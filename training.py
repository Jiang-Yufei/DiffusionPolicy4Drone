
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

from dataset import create_sample_indices, sample_sequence, get_data_stats, normalize_data, unnormalize_data, PushTImageDataset
from VisionEncoder import get_resnet, replace_submodules, replace_bn_with_gn
from network import SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock, ConditionalResidualBlock1D, ConditionalUnet1D


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
        with open(os.path.join(os.path.dirname(self.root_folder), 'config', 'training_config.json')) as json_file:
            self.config = json.load(json_file)
    
    def prepare_data(self):
            ids_path = os.path.join(self.args.data_root, self.args.env_id)
            with open(ids_path) as f:
                self.env_list = [line.rstrip() for line in f.readlines()]

            depth_transform = transforms.Compose([
                transforms.Resize((self.args.crop_size)),
                transforms.ToTensor()])
            
            total_img_data = 0
            track_id = 0
            test_env_id = min(self.args.test_env_id, len(self.env_list)-1)
            
            self.train_loader_list = []
            self.val_loader_list   = []
            

            for env_name in tqdm(self.env_list):
                if not self.args.training and track_id != test_env_id:
                    track_id += 1
                    continue
                
                data_path = os.path.join(*[self.args.data_root, self.args.env_type, env_name])

                train_data = PlannerData(root=data_path,
                                        train=True, 
                                        transform=depth_transform,
                                        goal_step=self.args.goal_step,
                                        max_episode=self.args.max_episode,
                                        max_depth=self.args.max_camera_depth)
                
                total_img_data += len(train_data)
                train_loader = MultiEpochsDataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
                
                self.train_loader_list.append(train_loader)

                val_data = PlannerData(root=data_path,
                                    train=False,
                                    transform=depth_transform,
                                    goal_step=self.args.goal_step,
                                    max_episode=self.args.max_episode,
                                    max_depth=self.args.max_camera_depth)

                val_loader = MultiEpochsDataLoader(val_data, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
                self.val_loader_list.append(val_loader)
                
            print("Data Loading Completed!")
            print("Number of image: %d | Number of goal-image pairs: %d"%(total_img_data, total_img_data * (int)(self.args.max_episode / self.args.goal_step)))
            
            return None 
            
    def parse_args(self):
        parser = argparse.ArgumentParser(description='Training script for PlannerNet')

        # dataConfig
        parser.add_argument("--data-root", type=str, default=os.path.join(self.root_folder, self.config['dataConfig'].get('data-root')), help="dataset root folder")
        parser.add_argument('--env-id', type=str, default=self.config['dataConfig'].get('env-id'), help='environment id list')
        parser.add_argument('--env_type', type=str, default=self.config['dataConfig'].get('env_type'), help='the dataset type')
        parser.add_argument('--crop-size', nargs='+', type=int, default=self.config['dataConfig'].get('crop-size'), help='image crop size')
        parser.add_argument('--max-camera-depth', type=float, default=self.config['dataConfig'].get('max-camera-depth'), help='maximum depth detection of camera, unit: meter')

        # modelConfig
        parser.add_argument("--model-save", type=str, default=os.path.join(self.root_folder, self.config['modelConfig'].get('model-save')), help="model save point")
        parser.add_argument('--resume', type=str, default=self.config['modelConfig'].get('resume'))
        parser.add_argument('--in-channel', type=int, default=self.config['modelConfig'].get('in-channel'), help='goal input channel numbers')
        parser.add_argument("--knodes", type=int, default=self.config['modelConfig'].get('knodes'), help="number of max nodes predicted")
        parser.add_argument("--goal-step", type=int, default=self.config['modelConfig'].get('goal-step'), help="number of frames betwen goals")
        parser.add_argument("--max-episode", type=int, default=self.config['modelConfig'].get('max-episode-length'), help="maximum episode frame length")

        # trainingConfig
        parser.add_argument('--training', type=str, default=self.config['trainingConfig'].get('training'))
        parser.add_argument("--lr", type=float, default=self.config['trainingConfig'].get('lr'), help="learning rate")
        parser.add_argument("--factor", type=float, default=self.config['trainingConfig'].get('factor'), help="ReduceLROnPlateau factor")
        parser.add_argument("--min-lr", type=float, default=self.config['trainingConfig'].get('min-lr'), help="minimum lr for ReduceLROnPlateau")
        parser.add_argument("--patience", type=int, default=self.config['trainingConfig'].get('patience'), help="patience of epochs for ReduceLROnPlateau")
        parser.add_argument("--epochs", type=int, default=self.config['trainingConfig'].get('epochs'), help="number of training epochs")
        parser.add_argument("--batch-size", type=int, default=self.config['trainingConfig'].get('batch-size'), help="number of minibatch size")
        parser.add_argument("--w-decay", type=float, default=self.config['trainingConfig'].get('w-decay'), help="weight decay of the optimizer")
        parser.add_argument("--num-workers", type=int, default=self.config['trainingConfig'].get('num-workers'), help="number of workers for dataloader")
        parser.add_argument("--gpu-id", type=int, default=self.config['trainingConfig'].get('gpu-id'), help="GPU id")

        # logConfig
        parser.add_argument("--log-save", type=str, default=os.path.join(self.root_folder, self.config['logConfig'].get('log-save')), help="train log file")
        parser.add_argument('--test-env-id', type=int, default=self.config['logConfig'].get('test-env-id'), help='the test env id in the id list')
        parser.add_argument('--visual-number', type=int, default=self.config['logConfig'].get('visual-number'), help='number of visualized trajectories')

        # sensorConfig
        parser.add_argument('--camera-tilt', type=float, default=self.config['sensorConfig'].get('camera-tilt'), help='camera tilt angle for visualization only')
        parser.add_argument('--sensor-offsetX-ANYmal', type=float, default=self.config['sensorConfig'].get('sensor-offsetX-ANYmal'), help='anymal front camera sensor offset in X axis')
        parser.add_argument("--fear-ahead-dist", type=float, default=self.config['sensorConfig'].get('fear-ahead-dist'), help="fear lookahead distance")

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
        self.pred_horizon = 32
        self.obs_horizon = 1
        self.action_horizon = 8
        # ResNet18 has output dim of 512
        self.vision_feature_dim = 512
        # agent_pos is 2 dimensional
        self.lowdim_obs_dim = 3
        self.goal_dim = 3
        # observation feature has 514 dims in total per step
        self.obs_dim = self.vision_feature_dim + self.goal_dim
        self.action_dim = 3

        # vision encoder
        self.vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder).to(self.args.gpu_id)
        # create network object
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon
        ).to(self.args.gpu_id)

        # the final arch has 2 parts
        self.nets = nn.ModuleDict({
            'vision_encoder': self.vision_encoder,
            'noise_pred_net': self.noise_pred_net
        }).to(self.args.gpu_id)

        self.optimizer = torch.optim.AdamW(params=self.nets.parameters(),lr=1e-4, weight_decay=1e-6)

        self.num_diffusion_iters = 100

        # Calculate total steps
        steps_per_epoch = sum(len(loader) for loader in self.train_loader_list)
        self.num_epochs = self.args.epochs
        total_steps = steps_per_epoch * self.num_epochs

        # Build LR scheduler
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=total_steps
        )


    def train(self):
        self.ema = EMAModel(
            parameters=self.nets.parameters(),
            power=0.75)
        
        self.noise_pred_net = self.nets['noise_pred_net']

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=False,
            prediction_type='epsilon'
        )

        with tqdm(range(self.num_epochs), desc='Epoch') as tglobal:
            for epoch_idx in tglobal:
                epoch_loss = []

                # Use train_loader_list as in test()
                combined = self.train_loader_list.copy()
                random.shuffle(combined)

                for env_id, loader in enumerate(combined):
                    enumerater = tqdm(enumerate(loader), desc=f"Env {env_id}", leave=False)

                    for batch_idx, inputs in enumerater:
                        # === Load and transfer to GPU ===
                        image = inputs[0].cuda(self.args.gpu_id)                       
                        # odom  = inputs[1].cuda(self.args.gpu_id)[...,0:3]
                        goal  = inputs[2].cuda(self.args.gpu_id)[...,0:3]
                        traj  = inputs[3].cuda(self.args.gpu_id)[...,0:3]
                         # (B, traj_len, 3)

                        # === Format batch like in test() ===
                        B = image.shape[0]
                        image = image.unsqueeze(1).expand(-1, self.obs_horizon, -1, -1, -1)        
                        # agent_pos = odom.unsqueeze(1).expand(-1, self.obs_horizon, -1)     
                        agent_goal = goal.unsqueeze(1).expand(-1, self.obs_horizon, -1)      

                        image_features = self.nets['vision_encoder'](image.flatten(end_dim=1))   
                        image_features = image_features.reshape(B, self.obs_horizon, -1)          

                        obs = torch.cat([image_features, agent_goal], dim=-1)                      
                        obs_cond = obs.flatten(start_dim=1)                                     

                        # === Diffusion training ===
                        noise = torch.randn((B, self.pred_horizon, self.action_dim), device=traj.device)

                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (B,), device=traj.device
                        ).long()

                        noisy_actions = self.noise_scheduler.add_noise(traj[:, :self.pred_horizon, :self.action_dim], noise, timesteps)
                        # print("before:", traj)
                        # print("after:", noisy_actions)
                        noise_pred = self.noise_pred_net(
                            sample=noisy_actions,
                            timestep=timesteps,
                            global_cond=obs_cond
                        )

                        # === Loss and optimize ===
                        loss = nn.functional.mse_loss(noise_pred, noise)

                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                        self.ema.step(self.nets.parameters())

                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        enumerater.set_postfix(loss=loss_cpu)

                tglobal.set_postfix(loss=np.mean(epoch_loss))

        self.ema.copy_to(self.nets.parameters())

        # Save the model
        torch.save(self.nets.state_dict(), self.args.model_save)

        return None
        
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

        # Replace BatchNorm with GroupNorm
        ema_nets['vision_encoder'] = replace_bn_with_gn(ema_nets['vision_encoder']).to(self.args.gpu_id)

        # Load model weights
        print(f"Loading model weights from {self.args.model_save}")
        state_dict = torch.load(self.args.model_save, map_location=f"cuda:{self.args.gpu_id}")
        ema_nets.load_state_dict(state_dict)

        # Wrap with EMA weights
        self.ema = EMAModel(parameters=ema_nets.parameters(), power=0.75)
        self.ema.copy_to(ema_nets.parameters())  # Transfer EMA weights into the new model
        ema_nets.eval()

        # Diffusion scheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=False,
            prediction_type='epsilon'
        )

        print("Running validation...")
        with torch.no_grad():
            for env_id, loader in enumerate(self.val_loader_list):
                for batch_idx, inputs in enumerate(loader):
                    image = inputs[0].cuda(self.args.gpu_id)
                    # odom  = inputs[1].cuda(self.args.gpu_id)[..., :3]
                    goal  = inputs[2].cuda(self.args.gpu_id)[..., :3]
                    traj  = inputs[3].cuda(self.args.gpu_id)[..., :3]  # (B, traj_len, 3)

                    B = image.shape[0]
                    image = image.unsqueeze(1).expand(-1, self.obs_horizon, -1, -1, -1)
                    # agent_pos = odom.unsqueeze(1).expand(-1, self.obs_horizon, -1)
                    agent_goal = goal.unsqueeze(1).expand(-1, self.obs_horizon, -1)

                    image_features = ema_nets['vision_encoder'](image.flatten(end_dim=1))
                    image_features = image_features.reshape(B, self.obs_horizon, -1)
                    obs = torch.cat([image_features, agent_goal], dim=-1)
                    obs_cond = obs.flatten(start_dim=1)

                    noisy_action = torch.randn((B, self.pred_horizon, self.action_dim), device=image.device)
                    naction = noisy_action

                    noise_scheduler.set_timesteps(self.num_diffusion_iters)

                    for k in noise_scheduler.timesteps:
                        noise_pred = ema_nets['noise_pred_net'](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                    # Convert and plot results
                    naction = naction.detach().cpu().numpy()
                    traj_gt = traj.detach().cpu().numpy()

                    for i in range(min(4, B)):
                        fig = plt.figure(figsize=(6, 6))
                        # Plot 3D trajectory
                        plt3d = fig.add_subplot(111)
                        plt3d = plt.axes(projection='3d')
                        plt3d.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1
                        plt3d.set_title(f"Env {env_id} | Batch {batch_idx} | Sample {i}")
                        plt3d.set_xlabel("X")
                        plt3d.set_ylabel("Y")
                        plt3d.set_zlabel("Z")

                        # Plot the trajectory
                        plt3d.plot(traj_gt[i, :, 0], traj_gt[i, :, 1], traj_gt[i, :, 2], label='Ground Truth', linewidth=2)
                        plt3d.plot(naction[i, :, 0], naction[i, :, 1], naction[i, :, 2], label='Predicted', linestyle='--', linewidth=2)
                        plt3d.scatter(0, 0, 0, color='green', label='Start')
                        plt3d.scatter(goal[i, 0].cpu(), goal[i, 1].cpu(), goal[i, 2].cpu(), color='red', label='Goal')
                        plt3d.legend()
                        plt3d.grid(True)
                        plt3d.view_init(elev=20, azim=-35)
                        plt.tight_layout()
                        plt.savefig(f"val_vis/env{env_id}_batch{batch_idx}_sample_{i}.png")
                        plt.close()
            print(f"Validation done for env {env_id}")

if __name__ == "__main__":
    trainer = PlannerNetTrainer()
    # trainer.test()
    # trainer.train()
    trainer.validate()