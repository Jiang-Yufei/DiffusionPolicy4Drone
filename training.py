
import os
import PIL
import torch
import torch.nn as nn
import numpy as np
import pypose as pp
from PIL import Image
from pathlib import Path
from random import sample
from operator import itemgetter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import tqdm
import json
import random
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import MultiEpochsDataLoader
import argparse

from VisionEncoder import get_resnet, replace_submodules, replace_bn_with_gn
from network import SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock, ConditionalResidualBlock1D, ConditionalUnet1D
import imageio


class PlannerData(Dataset):
    def __init__(self, root, max_episode, goal_step, train, ratio=0.9, max_depth=10.0, sensorOffsetX=0.0, transform=None, is_robot=True, min_distance=0.5):

        super().__init__()
        self.transform = transform
        self.is_robot = is_robot
        self.max_depth = max_depth
        img_filename_list = []
        img_path = os.path.join(root, "depth")
        img_filename_list = [str(s) for s in Path(img_path).rglob('*.png')]

        #img_filename_list.sort(key = lambda x : int(x.split("/")[-1][:-4]))
        img_filename_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

        odom_path = os.path.join(root, "odom_ground_truth.txt")
        odom_list = []
        offset_T = pp.identity_SE3()
        offset_T.tensor()[0] = sensorOffsetX
        with open(odom_path) as f:
            lines = f.readlines()
            for line in lines:
                odom = np.fromstring(line[1:-2], dtype=np.float32, sep=', ')
                odom = pp.SE3(odom)
                if is_robot:
                    odom = odom @ offset_T
                odom_list.append(odom)

        N = len(odom_list)

        self.img_filename = []
        self.odom_list    = []
        self.goal_list    = []
        self.traj_list    = []
        self.traj_length = 32

        for ahead in range(0, max_episode, goal_step):
            for i in range(N-1):
                odom = odom_list[i]
                goal = odom_list[min(i+self.traj_length, N-1)]
                goal = (pp.Inv(odom) @ goal)
                    
                traj = odom_list[i:min([i+self.traj_length, N-1])]
                traj = [pp.Inv(odom) @ t for t in traj]

                traj = torch.stack(traj) # shape: (32, 7)
                # print(traj.shape)

                if traj.shape[0] < self.traj_length:
                    continue  # Skip this sample

                self.img_filename.append(img_filename_list[i])
                self.odom_list.append(odom.tensor())
                self.goal_list.append(goal.tensor())
                self.traj_list.append(torch.tensor(traj, dtype=torch.float32))


        N = len(self.odom_list)

        indexfile = os.path.join(img_path, 'split.pt')
        is_generate_split = True
        if os.path.exists(indexfile):
            train_index, test_index = torch.load(indexfile, weights_only=False)
            if len(train_index)+len(test_index) == N:
                is_generate_split = False
            else:
                print("Data changed! Generate a new split file")
        if (is_generate_split):
            indices = range(N)
            train_index = sample(indices, int(ratio*N))
            test_index = np.delete(indices, train_index)
            torch.save((train_index, test_index), indexfile)

        if train == True:
            self.img_filename = itemgetter(*train_index)(self.img_filename)
            self.odom_list    = itemgetter(*train_index)(self.odom_list)
            self.goal_list    = itemgetter(*train_index)(self.goal_list)
            self.traj_list    = itemgetter(*train_index)(self.traj_list)

            while len(self.traj_list) > 0 and self.traj_list[-1].shape[0] != self.traj_length:
                self.img_filename = self.img_filename[:-1]
                self.odom_list    = self.odom_list[:-1]
                self.goal_list    = self.goal_list[:-1]
                self.traj_list    = self.traj_list[:-1]

        else:
            self.img_filename = itemgetter(*test_index)(self.img_filename)
            self.odom_list    = itemgetter(*test_index)(self.odom_list)
            self.goal_list    = itemgetter(*test_index)(self.goal_list)
            self.traj_list    = itemgetter(*test_index)(self.traj_list)
            
            while len(self.traj_list) > 0 and self.traj_list[-1].shape[0] != self.traj_length:
                self.img_filename = self.img_filename[:-1]
                self.odom_list    = self.odom_list[:-1]
                self.goal_list    = self.goal_list[:-1]
                self.traj_list    = self.traj_list[:-1]

        assert len(self.odom_list) == len(self.img_filename), "odom numbers should match with image numbers"
        for traj in self.traj_list:
            assert len(traj) == self.traj_length, "traj length shouldn't be %s" % (traj.shape,)
        

    def __len__(self):
        return len(self.img_filename)

    def __getitem__(self, idx):
        image = Image.open(self.img_filename[idx])
        if self.is_robot:
            image = np.array(image.transpose(PIL.Image.ROTATE_180))
        else:
            image = np.array(image)
        image[~np.isfinite(image)] = 0.0
        image = (image / 1000.0).astype("float32")
        image[image > self.max_depth] = 0.0
        # DEBUG show image
        # img = Image.fromarray((image * 255 / np.max(image)).astype('uint8'))
        # img.show()
        image = Image.fromarray(image)
        image = self.transform(image).expand(3, -1, -1)

        data = (image, self.odom_list[idx], self.goal_list[idx], self.traj_list[idx])

        # assert wrong dims and print dims
        assert self.odom_list[idx].shape == (7,)
        assert self.goal_list[idx].shape == (7,)
        assert self.traj_list[idx].shape == (self.traj_length, 7), "got %s when idx is %s" % (self.traj_list[idx].shape,idx)

        return data

class PlannerNetTrainer():
    def __init__(self):
        self.root_folder = '.'
        self.load_config()
        self.parse_args()
        self.prepare_data()
        self.prepare_model()

    def load_config(self):
        cfg_path = os.path.join(os.path.dirname(self.root_folder), 'config', 'training_config.json')
        with open(cfg_path, 'r') as f:
            self.config = json.load(f)
    
    def prepare_data(self):
        ids_path = os.path.join(self.args.data_root, self.args.env_id)
        with open(ids_path) as f:
            self.env_list = [line.rstrip() for line in f.readlines()]

        depth_transform = transforms.Compose([
            transforms.Resize((self.args.crop_size)),
            transforms.ToTensor()
        ])

        total_img_data = 0
        track_id = 0
        test_env_id = min(self.args.test_env_id, len(self.env_list) - 1)

        self.train_loader_list = []
        self.val_loader_list = []

        for env_name in tqdm(self.env_list):
            if not self.args.training and track_id != test_env_id:
                track_id += 1
                continue

            data_path = os.path.join(*[self.args.data_root, self.args.env_type, env_name])

            train_data = PlannerData(
                root=data_path,
                train=True,
                transform=depth_transform,
                goal_step=self.args.goal_step,
                max_episode=self.args.max_episode,
                ratio=self.args.train_split_ratio,
                max_depth=self.args.max_camera_depth,
                sensorOffsetX=self.args.sensor_offset_x,
                is_robot=self.args.is_robot,
            )
            total_img_data += len(train_data)
            train_loader = MultiEpochsDataLoader(
                train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers
            )
            self.train_loader_list.append(train_loader)

            val_data = PlannerData(
                root=data_path,
                train=False,
                transform=depth_transform,
                goal_step=self.args.goal_step,
                max_episode=self.args.max_episode,
                ratio=self.args.train_split_ratio,
                max_depth=self.args.max_camera_depth,
                sensorOffsetX=self.args.sensor_offset_x,
                is_robot=self.args.is_robot,
            )
            val_loader = MultiEpochsDataLoader(
                val_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers
            )
            self.val_loader_list.append(val_loader)

        print("Data Loading Completed!")
        print("Number of image: %d | Number of goal-image pairs: %d" % (
            total_img_data, total_img_data * int(self.args.max_episode / self.args.goal_step)
        ))
          
    def parse_args(self):
        parser = argparse.ArgumentParser(description='Training script for DiffusionPolicy4Drone')

        cfg = self.config

        # ---------- data ----------
        parser.add_argument("--data-root", type=str, default=os.path.join(self.root_folder, cfg["data"]["data_root"]))
        parser.add_argument("--env-id", type=str, default=cfg["data"]["env_id"])
        parser.add_argument("--env-type", type=str, default=cfg["data"]["env_type"])
        parser.add_argument("--crop-size", nargs='+', type=int, default=cfg["data"]["crop_size"])
        parser.add_argument("--max-camera-depth", type=float, default=cfg["data"]["max_camera_depth"])
        parser.add_argument("--train-split-ratio", type=float, default=cfg["data"].get("train_split_ratio", 0.9))
        parser.add_argument("--num-workers", type=int, default=cfg["data"].get("num_workers", 4))

        # ---------- task / sampling ----------
        parser.add_argument("--goal-step", type=int, default=cfg["task"]["goal_step"])
        parser.add_argument("--max-episode", type=int, default=cfg["task"]["max_episode_length"])
        parser.add_argument("--sensor-offset-x", type=float, default=cfg["task"].get("sensor_offset_x", 0.0))
        parser.add_argument("--is-robot", action="store_true" if cfg["task"].get("is_robot", True) else "store_false")

        # ---------- diffusion ----------
        parser.add_argument("--pred-horizon", type=int, default=cfg["diffusion"]["pred_horizon"])
        parser.add_argument("--obs-horizon", type=int, default=cfg["diffusion"]["obs_horizon"])
        parser.add_argument("--action-dim", type=int, default=cfg["diffusion"]["action_dim"])
        parser.add_argument("--num-train-timesteps", type=int, default=cfg["diffusion"]["num_train_timesteps"])
        parser.add_argument("--beta-schedule", type=str, default=cfg["diffusion"]["beta_schedule"])
        parser.add_argument("--prediction-type", type=str, default=cfg["diffusion"]["prediction_type"])
        parser.add_argument("--ema-power", type=float, default=cfg["diffusion"]["ema_power"])
        parser.add_argument("--warmup-steps", type=int, default=cfg["diffusion"].get("warmup_steps", 500))
        parser.add_argument("--lr-schedule", type=str, default=cfg["diffusion"].get("lr_schedule", "cosine"))

        # ---------- training ----------
        parser.add_argument("--training", type=bool, default=cfg["train"]["training"])
        parser.add_argument("--epochs", type=int, default=cfg["train"]["epochs"])
        parser.add_argument("--batch-size", type=int, default=cfg["train"]["batch_size"])
        parser.add_argument("--lr", type=float, default=cfg["train"]["lr"])
        parser.add_argument("--w-decay", type=float, default=cfg["train"]["weight_decay"])
        parser.add_argument("--gpu-id", type=int, default=cfg["train"]["gpu_id"])
        parser.add_argument("--seed", type=int, default=cfg["train"].get("seed", 0))

        # ---------- checkpoint / eval ----------
        parser.add_argument("--model-save", type=str, default=os.path.join(self.root_folder, cfg["checkpoint"]["model_save"]))
        parser.add_argument("--resume", type=bool, default=cfg["checkpoint"].get("resume", False))
        parser.add_argument("--test-env-id", type=int, default=cfg["eval"].get("test_env_id", 0))

        self.args = parser.parse_args()

    def test(self):

        env_num = len(self.train_loader_list)
        
        # Zip the lists and convert to a list of tuples
        combined = self.train_loader_list.copy()
        random.shuffle(combined)


        # Iterate through shuffled pairs
        for env_id, loader in enumerate(combined):
            enumerater = tqdm.tqdm(enumerate(loader))
            for batch_idx, inputs in enumerater:
                if torch.cuda.is_available():
                    image = inputs[0].cuda(self.args.gpu_id)
                    # odom  = inputs[1].cuda(self.args.gpu_id)
                    goal  = inputs[2].cuda(self.args.gpu_id)
                    traj  = inputs[3].cuda(self.args.gpu_id)[0:3]


            #################################################################
            pred_horizon = self.pred_horizon
            obs_horizon = self.obs_horizon

            vision_encoder = get_resnet('resnet18').to(self.args.gpu_id)

            # IMPORTANT!
            # replace all BatchNorm with GroupNorm to work with EMA
            # performance will tank if you forget to do this!
            vision_encoder = replace_bn_with_gn(vision_encoder).to(self.args.gpu_id)

            # ResNet18 has output dim of 512
            vision_feature_dim = self.vision_feature_dim
            # agent_pos is 2 dimensional
            lowdim_obs_dim = self.lowdim_obs_dim
            # observation feature has 514 dims in total per step
            obs_dim = self.obs_dim
            action_dim = self.action_dim

            # create network object
            noise_pred_net = ConditionalUnet1D(
                input_dim=action_dim,
                global_cond_dim=obs_dim*obs_horizon
            ).to(self.args.gpu_id)

            # the final arch has 2 parts
            nets = nn.ModuleDict({
                'vision_encoder': vision_encoder,
                'noise_pred_net': noise_pred_net
            }).to(self.args.gpu_id)

            # demo
            with torch.no_grad():
                # example inputs
                agent_pos = goal.unsqueeze(1).expand(-1, obs_horizon, -1)[...,0:3]
                image = image.unsqueeze(1).expand(-1, obs_horizon, -1, -1, -1)
                # vision encoder
                image_features = nets['vision_encoder']( # 64 2 3 96 96
                    image.flatten(end_dim=1))
                image_features = image_features.reshape(*image.shape[:2],-1)
                obs = torch.cat([image_features, agent_pos],dim=-1)
                # print("Observation shape: ", obs.shape)
                # print("Observation feature shape: ", obs.flatten(start_dim=1).shape)

                noised_action = torch.randn((4, pred_horizon, action_dim)).to(self.args.gpu_id)
                diffusion_iter = torch.zeros((4,)).to(self.args.gpu_id)

                # the noise prediction network
                # takes noisy action, diffusion iteration and observation as input
                # predicts the noise added to action
                noise = nets['noise_pred_net'](
                    sample=noised_action,
                    timestep=diffusion_iter,
                    global_cond=obs.flatten(start_dim=1))

                # illustration of removing noise
                # the actual noise removal is performed by NoiseScheduler
                # and is dependent on the diffusion noise schedule
                denoised_action = noised_action - noise

            # for this demo, we use DDPMScheduler with 100 diffusion iterations
            num_diffusion_iters = 100
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_diffusion_iters,
                # the choise of beta schedule has big impact on performance
                # we found squared cosine works the best
                beta_schedule='squaredcos_cap_v2',
                # clip output to [-1,1] to improve stability
                clip_sample=True,
                # our network predicts noise (instead of denoised action)
                prediction_type='epsilon'
            )

            # device transfer
            device = torch.device('cuda')
            _ = nets.to(device)

    def prepare_model(self):
        # horizons / dims
        self.pred_horizon = int(self.args.pred_horizon)
        self.obs_horizon  = int(self.args.obs_horizon)
        self.action_dim   = int(self.args.action_dim)

        # vision encoder dims (ResNet18)
        self.vision_feature_dim = 512
        self.goal_dim = 3
        self.obs_dim = self.vision_feature_dim + self.goal_dim  # 512 + 3

        # diffusion iters
        self.num_diffusion_iters = int(self.args.num_train_timesteps)

        # vision encoder
        self.vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder).to(self.args.gpu_id)

        # noise prediction net
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim * self.obs_horizon
        ).to(self.args.gpu_id)

        self.nets = nn.ModuleDict({
            'vision_encoder': self.vision_encoder,
            'noise_pred_net': self.noise_pred_net
        }).to(self.args.gpu_id)

        self.optimizer = torch.optim.AdamW(
            params=self.nets.parameters(),
            lr=float(self.args.lr),
            weight_decay=float(self.args.w_decay)
        )

        # total steps for LR scheduler
        steps_per_epoch = sum(len(loader) for loader in self.train_loader_list)
        self.num_epochs = int(self.args.epochs)
        total_steps = steps_per_epoch * self.num_epochs

        self.lr_scheduler = get_scheduler(
            name=str(self.args.lr_schedule),
            optimizer=self.optimizer,
            num_warmup_steps=int(self.args.warmup_steps),
            num_training_steps=int(total_steps)
        )


    def train(self):
        self.ema = EMAModel(parameters=self.nets.parameters(), power=float(self.args.ema_power))
        self.noise_pred_net = self.nets['noise_pred_net']

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(self.args.num_train_timesteps),
            beta_schedule=str(self.args.beta_schedule),
            clip_sample=False,
            prediction_type=str(self.args.prediction_type)
        )

        all_epoch_losses = []

        with tqdm(range(self.num_epochs), desc='Epoch') as tglobal:
            for epoch_idx in tglobal:
                epoch_loss = []
                combined = self.train_loader_list.copy()
                random.shuffle(combined)

                for env_id, loader in enumerate(combined):
                    enumerater = tqdm(enumerate(loader), desc=f"Env {env_id}", leave=False)

                    for batch_idx, inputs in enumerater:
                        image = inputs[0].cuda(self.args.gpu_id)
                        goal  = inputs[2].cuda(self.args.gpu_id)[..., 0:3]
                        traj  = inputs[3].cuda(self.args.gpu_id)[..., 0:3]

                        B = image.shape[0]

                        image = image.unsqueeze(1).expand(-1, self.obs_horizon, -1, -1, -1)
                        agent_goal = goal.unsqueeze(1).expand(-1, self.obs_horizon, -1)

                        image_features = self.nets['vision_encoder'](image.flatten(end_dim=1))
                        image_features = image_features.reshape(B, self.obs_horizon, -1)

                        obs = torch.cat([image_features, agent_goal], dim=-1)
                        obs_cond = obs.flatten(start_dim=1)

                        noise = torch.randn((B, self.pred_horizon, self.action_dim), device=traj.device)
                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (B,), device=traj.device
                        ).long()

                        noisy_actions = self.noise_scheduler.add_noise(
                            traj[:, :self.pred_horizon, :self.action_dim],
                            noise,
                            timesteps
                        )

                        noise_pred = self.noise_pred_net(
                            sample=noisy_actions,
                            timestep=timesteps,
                            global_cond=obs_cond
                        )

                        loss = nn.functional.mse_loss(noise_pred, noise)

                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                        self.ema.step(self.nets.parameters())

                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        enumerater.set_postfix(loss=loss_cpu)

                epoch_avg_loss = float(np.mean(epoch_loss)) if len(epoch_loss) else 0.0
                all_epoch_losses.append(epoch_avg_loss)
                tglobal.set_postfix(loss=epoch_avg_loss)

        self.ema.copy_to(self.nets.parameters())

        torch.save(self.nets.state_dict(), self.args.model_save)

        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "training_loss.txt"), "w") as f:
            for i, loss in enumerate(all_epoch_losses):
                f.write(f"{i}\t{loss:.6f}\n")

        plt.figure(figsize=(8, 6))
        plt.plot(all_epoch_losses, label="Training Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss Over Epochs")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("logs/training_loss.png")
        plt.close()


    def validate(self):
        print("Loading trained model for validation...")

        # Create EMA nets with same architecture
        ema_nets = nn.ModuleDict({
            'vision_encoder': get_resnet('resnet18'),
            'noise_pred_net': ConditionalUnet1D(
                input_dim=self.action_dim,
                global_cond_dim=self.obs_dim * self.obs_horizon
            )
        }).to(self.args.gpu_id)

        ema_nets['vision_encoder'] = replace_bn_with_gn(ema_nets['vision_encoder']).to(self.args.gpu_id)

        print(f"Loading model weights from {self.args.model_save}")
        state_dict = torch.load(self.args.model_save, map_location=f"cuda:{self.args.gpu_id}")
        ema_nets.load_state_dict(state_dict)

        self.ema = EMAModel(parameters=ema_nets.parameters(), power=float(self.args.ema_power))
        self.ema.copy_to(ema_nets.parameters())
        ema_nets.eval()

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(self.args.num_train_timesteps),
            beta_schedule=str(self.args.beta_schedule),
            clip_sample=False,
            prediction_type=str(self.args.prediction_type)
        )

        print("Running validation...")
        with torch.no_grad():
            for env_id, loader in enumerate(self.val_loader_list):
                for batch_idx, inputs in enumerate(loader):
                    image = inputs[0].cuda(self.args.gpu_id)
                    goal = inputs[2].cuda(self.args.gpu_id)[..., :3]
                    traj = inputs[3].cuda(self.args.gpu_id)[..., :3]

                    B = image.shape[0]
                    image = image.unsqueeze(1).expand(-1, self.obs_horizon, -1, -1, -1)
                    agent_goal = goal.unsqueeze(1).expand(-1, self.obs_horizon, -1)

                    image_features = ema_nets['vision_encoder'](image.flatten(end_dim=1))
                    image_features = image_features.reshape(B, self.obs_horizon, -1)
                    obs = torch.cat([image_features, agent_goal], dim=-1)
                    obs_cond = obs.flatten(start_dim=1)

                    noisy_action = torch.randn((B, self.pred_horizon, self.action_dim), device=image.device)
                    naction = noisy_action
                    noise_scheduler.set_timesteps(self.num_diffusion_iters)

                    # Save intermediate steps
                    step_interval = self.num_diffusion_iters // 30  # finer steps for smoother gif
                    intermediate_samples = []

                    for k in noise_scheduler.timesteps:
                        noise_pred = ema_nets['noise_pred_net'](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        step_result = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        )
                        naction = step_result.prev_sample

                        if k % step_interval == 0 or k == noise_scheduler.timesteps[-1]:
                            intermediate_samples.append(naction.detach().cpu().clone())

                    traj_gt = traj.detach().cpu().numpy()
                    os.makedirs("val_vis/diffusion_steps", exist_ok=True)

                    for i in range(min(1, B)):  # Just show one sample per batch
                        gif_frames = []
                        for j, traj_tensor in enumerate(intermediate_samples):
                            fig = plt.figure(figsize=(6, 6))
                            ax = fig.add_subplot(111, projection='3d')
                            ax.set_title(f"Step {j * step_interval}")
                            traj_np = traj_tensor[i].numpy()

                            ax.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2], linestyle='--', label='Predicted')
                            ax.plot(traj_gt[i, :, 0], traj_gt[i, :, 1], traj_gt[i, :, 2], label='Ground Truth')
                            ax.scatter(0, 0, 0, color='green', label='Start')
                            ax.scatter(goal[i, 0].cpu(), goal[i, 1].cpu(), goal[i, 2].cpu(), color='red', label='Goal')
                            ax.set_box_aspect([1, 1, 1])
                            ax.legend()
                            ax.grid(True)
                            ax.view_init(elev=20, azim=-35)

                            frame_path = f"val_vis/diffusion_steps/tmp_frame_{j}.png"
                            plt.savefig(frame_path)
                            gif_frames.append(frame_path)
                            plt.close()

                        gif_out_path = f"val_vis/diffusion_steps/env{env_id}_batch{batch_idx}_sample{i}_denoising.gif"
                        images = [imageio.imread(f) for f in gif_frames]
                        imageio.mimsave(gif_out_path, images, duration=0.5)

                        # Clean up temp frames
                        for f in gif_frames:
                            os.remove(f)

                print(f"Validation done for env {env_id}")

if __name__ == "__main__":
    trainer = PlannerNetTrainer()
    # trainer.test()
    trainer.train()
    # trainer.validate()