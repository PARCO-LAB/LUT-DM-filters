import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

torch.set_printoptions(sci_mode=False)

import os
import logging
import time
import glob
import argparse

import os.path as path
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import pandas as pd

from models.gcndiff import GCNdiff, adj_mx_from_edges
from models.ema import EMAHelper

from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps, ddpm_steps
from common.loss import mpjpe, p_mpjpe

from torch.utils.data import TensorDataset, DataLoader

class Diffpose(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        # GraFormer mask
        #self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
        #                        True, True, True, True, True, True, True]]]).cuda()
        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                                True, True]]]).cuda()
        
        # Generate Diffusion sequence parameters
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def data_from_dataframe_12_xyz(self, df=None):
        #df = df.drop(columns="time")

        for i in df.columns:
            if 'RFoot' in i:
                df = df.drop(columns=[i])
            if 'LFoot' in i:
                df = df.drop(columns=[i])
            if 'Neck' in i:
                df = df.drop(columns=[i])
            if 'Head' in i:
                df = df.drop(columns=[i])

        for i in df.columns:
            if 'U' in i:
                df = df.drop(columns=[i])
            if 'V' in i:
                df = df.drop(columns=[i])

        indexes = []
        for i in df.columns:
            if 'X' in i:
                indexes.append(i) 
 
        # Convert pandas DataFrame to a PyTorch tensor
        tensor_from_csv = torch.tensor(df.values, dtype=torch.float32)

        tensor_from_csv = tensor_from_csv.reshape(-1, 12, 3)
 
        return tensor_from_csv

    def data_from_dataframe_12_uv(self, df):
        #df = df.drop(columns="time")

        for i in df.columns:
            if 'RFoot' in i:
                df = df.drop(columns=[i])
            if 'LFoot' in i:
                df = df.drop(columns=[i])
            if 'Neck' in i:
                df = df.drop(columns=[i])
            if 'Head' in i:
                df = df.drop(columns=[i])
            if 'Unnamed' in i:
                df = df.drop(columns=[i])

        for i in df.columns:
            if 'X' in i:
                df = df.drop(columns=[i])
            if 'Y' in i:
                df = df.drop(columns=[i])
            if 'Z' in i:
                df = df.drop(columns=[i])

        indexes = []
        for i in df.columns:
            if 'U' in i:
                indexes.append(i) 

        # Convert pandas DataFrame to a PyTorch tensor
        tensor_from_csv = torch.tensor(df.values, dtype=torch.float32)

        tensor_from_csv = tensor_from_csv.reshape(-1, 12, 2)

        return tensor_from_csv

    def read_csv_files_from_folder(self, folder_path, subjects):
            """
            Read one CSV file at a time from a folder.

            Parameters:
            - folder_path (str): The path to the folder containing CSV files.

            Returns:
            - list: A list of pandas DataFrames, each representing a CSV file.
            """
            csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv') and any(file.startswith(prefix+"_") for prefix in subjects)]
            
            concatenated_df = pd.DataFrame()
            for csv_file in csv_files:
                csv_file_path = os.path.join(folder_path, csv_file)
                dataframe = pd.read_csv(csv_file_path)

                concatenated_df = pd.concat([concatenated_df, dataframe], axis=0, ignore_index=True)

            #print(concatenated_df.head)
            concatenated_df /= 1000
            #print(concatenated_df)
            tensor_from_csv_xyz = self.data_from_dataframe_12_xyz(concatenated_df)
            tensor_from_csv_uv = self.data_from_dataframe_12_uv(concatenated_df)

            return tensor_from_csv_uv, tensor_from_csv_xyz

    def read_noisy_and_gt_csv_files_from_folder(self, folder_path, folder_path_gt, subjects):
            csv_files = [file for file in os.listdir(folder_path_gt) if file.endswith('.csv') and any(file.startswith(prefix+"_") for prefix in subjects)]
            
            csv_files.sort()
            #print(csv_files)

            concatenated_df = pd.DataFrame()
            concatenated_df_gt = pd.DataFrame()

            for csv_file in csv_files:
                csv_file_path = os.path.join(folder_path, csv_file)
                
                csv_file_path_gt = os.path.join(folder_path_gt, csv_file)

                dataframe = pd.read_csv(csv_file_path)
                dataframe_gt = pd.read_csv(csv_file_path_gt)

                #print(csv_file_path)

                if len(dataframe) == len(dataframe_gt):
                    if dataframe.isna().sum().sum() > 0:
                        print(str(dataframe.isna().sum().sum()) + " entries cancelled")
                        nan_indices_df1 = dataframe[dataframe.isna().any(axis=1)].index
                        df_cleaned = dataframe.dropna()
                        df_gt_cleaned = dataframe_gt.drop(nan_indices_df1)

                        if df_cleaned.isna().sum().sum() == 0:
                            concatenated_df = pd.concat([concatenated_df, df_cleaned], axis=0, ignore_index=True)
                            concatenated_df_gt = pd.concat([concatenated_df_gt, df_gt_cleaned], axis=0, ignore_index=True)
                        else:
                            print("Not correct")
                    else:
                        print(csv_file + "correct")
                        concatenated_df = pd.concat([concatenated_df, dataframe], axis=0, ignore_index=True)
                        concatenated_df_gt = pd.concat([concatenated_df_gt, dataframe_gt], axis=0, ignore_index=True)
                else:
                    print(csv_file + " has different lengths")
                    
            #print(concatenated_df.head)
            concatenated_df /= 1000
            concatenated_df_gt /= 1000
            #print(concatenated_df)
            tensor_from_csv_xyz = self.data_from_dataframe_12_xyz(concatenated_df)
            tensor_from_csv_uv = self.data_from_dataframe_12_uv(concatenated_df)

            tensor_from_csv_xyz_gt = self.data_from_dataframe_12_xyz(concatenated_df_gt)
            tensor_from_csv_uv_gt = self.data_from_dataframe_12_uv(concatenated_df_gt)

            return tensor_from_csv_uv, tensor_from_csv_xyz, tensor_from_csv_uv_gt, tensor_from_csv_xyz_gt

    def read_noisy_and_gt_csv_files_from_folder_trt(self, folder_path, folder_path_gt):
        args, config = self.args, self.config

        csv_files = []
        
        csv_files = [file for file in os.listdir(folder_path_gt) if file.endswith('.csv')]
            #csv_files = ['s1_walking1_cam1.csv']
        
        csv_files.sort()
        #print(csv_files)

        concatenated_df = pd.DataFrame()
        list_dat = []
        concatenated_df_gt = pd.DataFrame()

        for csv_file in csv_files:
            try:
                csv_file_path = os.path.join(folder_path, csv_file)
                
                csv_file_path_gt = os.path.join(folder_path_gt, csv_file)

                dataframe = pd.read_csv(csv_file_path)
                dataframe_gt = pd.read_csv(csv_file_path_gt)

                # Crop to the size of the shorter DataFrame along rows
                min_rows = min(dataframe.shape[0], dataframe_gt.shape[0])

                dataframe = dataframe.iloc[:min_rows, :]

                list_dat.append([csv_file]*len(dataframe))
                dataframe_gt = dataframe_gt.iloc[:min_rows, :]
                
            except:
                print(csv_file_path, csv_file_path_gt, csv_file)
                print("Not found!\n")
                continue

            #print(csv_file_path)
            
            concatenated_df = pd.concat([concatenated_df, dataframe], axis=0, ignore_index=True)
            concatenated_df_gt = pd.concat([concatenated_df_gt, dataframe_gt], axis=0, ignore_index=True)
            
        #print(concatenated_df.head)
        concatenated_df /= 1000
        concatenated_df_gt /= 1000
        #print(concatenated_df)
        tensor_from_csv_xyz = self.data_from_dataframe_12_xyz(concatenated_df)
        tensor_from_csv_uv = self.data_from_dataframe_12_uv(concatenated_df)

        tensor_from_csv_xyz_gt = self.data_from_dataframe_12_xyz(concatenated_df_gt)
        tensor_from_csv_uv_gt = self.data_from_dataframe_12_uv(concatenated_df_gt)

        flattened_list = [item for sublist in list_dat for item in sublist]

        return tensor_from_csv_uv, tensor_from_csv_xyz, tensor_from_csv_uv_gt, tensor_from_csv_xyz_gt, flattened_list

    # prepare 2D and 3D skeleton for model training and testing 
    def prepare_data_from_csv(self):
        args, config = self.args, self.config
        print('==> Using settings {}'.format(args))
        print('==> Using configures {}'.format(config))
        
        # load dataset
        if config.data.dataset == "human36m":
            from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
            
            self.subjects_train = TRAIN_SUBJECTS
            self.subjects_test = TEST_SUBJECTS
            
            folder_path = config.folder_path
            folder_path_gt = config.folder_path_gt

            train_input_2d, train_inputs_xyz, train_input_2d_gt, train_data_gt_xyz = self.read_noisy_and_gt_csv_files_from_folder(folder_path, folder_path_gt, self.subjects_train)
            test_input_2d, test_inputs_xyz, test_input_2d_gt, test_data_gt_xyz = self.read_noisy_and_gt_csv_files_from_folder(folder_path, folder_path_gt, self.subjects_test)
            
            if(torch.equal(train_input_2d, train_input_2d_gt)):
               print("EQ train")

            if(torch.equal(test_input_2d, test_input_2d_gt)):
               print("EQ test")

            # Zero-Hip Centered
            train_inputs_xyz[:, :, :] = train_inputs_xyz[:, :, :] - (train_inputs_xyz[:, :1, :] + train_inputs_xyz[:,3:4,:])/2
            test_inputs_xyz[:, :, :] = test_inputs_xyz[:, :, :] - (test_inputs_xyz[:, :1, :] + test_inputs_xyz[:,3:4,:])/2
            train_data_gt_xyz[:, :, :] = train_data_gt_xyz[:, :, :] - (train_data_gt_xyz[:, :1, :] + train_data_gt_xyz[:,3:4,:])/2
            test_data_gt_xyz[:, :, :] = test_data_gt_xyz[:, :, :] - (test_data_gt_xyz[:, :1, :] + test_data_gt_xyz[:,3:4,:])/2

            train_input_2d[:, :, :] = train_input_2d[:, :, :] - (train_input_2d[:, :1, :] + train_input_2d[:,3:4,:])/2
            test_input_2d[:, :, :] = test_input_2d[:, :, :] - (test_input_2d[:, :1, :] + test_input_2d[:,3:4,:])/2
            
            train_dataset = TensorDataset(train_inputs_xyz, train_input_2d, train_data_gt_xyz)
            test_dataset = TensorDataset(test_inputs_xyz, test_input_2d, test_data_gt_xyz)
            # Create a DataLoader
            batch_size = config.training.batch_size
            config.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            config.dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  

            epoch_loss_3d_pos = AverageMeter()
            epoch_loss_3d_pos_procrustes = AverageMeter()

            for i, (inputs_xyz, input_2d, targets_3d) in enumerate(config.dataloader):
                input_2d = input_2d.squeeze()
                inputs_xyz = inputs_xyz.squeeze()
                targets_3d = targets_3d.squeeze()

                epoch_loss_3d_pos.update(mpjpe(inputs_xyz, targets_3d).item() * 1000.0, targets_3d.size(0))
                #epoch_loss_3d_pos_procrustes.update(p_mpjpe(inputs_xyz.numpy(), targets_3d.numpy()).item() * 1000.0, targets_3d.size(0))\
            
            logging.info('Dataset | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                .format(e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg))
            
            return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg
        else:
            raise KeyError('Invalid dataset')

    # create diffusion model
    def create_diffusion_model(self, model_path = None):
        args, config = self.args, self.config
        # EDGES 12
        edges = torch.tensor([[0, 1], [0, 9],[1, 2], [0, 3], [3, 6], [3, 4], [4, 5],
                            [6, 7], [6, 9], [7, 8], [9, 10], [10, 11]], dtype=torch.long)
        
        #adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        adj = adj_mx_from_edges(num_pts=12, edges=edges, sparse=False)
        self.model_diff = GCNdiff(adj.cuda(), config).cuda()
        
        self.model_diff = torch.nn.DataParallel(self.model_diff)

        # load pretrained model
        if model_path:
            states = torch.load(model_path)
            self.model_diff.load_state_dict(states[0])
            
    def train(self):
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask

        import json
        json_file_path = args.log_path + '/loss.json'
        # initialize the recorded best performance
        best_p1, best_epoch1, best_epoch2 = 1000, 0, 0
        best_p3 = best_p1
        p1_list = []
        p3_list = []
        loss_list = []
        loss_mean_list = []
        # skip rate when sample skeletons from video
        stride = self.args.downsample
        
        dataloader = config.train_dataloader

        optimizer = get_optimizer(self.config, self.model_diff.parameters())
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
        else:
            ema_helper = None
        
        start_epoch, step = 0, 0
        
        lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma

        random = self.config.training.random

        if random == True: print("torch.randn_like(x)")
        else : print("noisy_input_uvxyz - targets_uvxyz")

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0

            # Switch to train mode
            torch.set_grad_enabled(True)
            self.model_diff.train()
            
            epoch_loss_diff = AverageMeter()

            for i, (inputs_xyz, input_2d, data_gt_xyz) in enumerate(dataloader):
                data_time += time.time() - data_start
                step += 1

                targets_uvxyz = torch.cat([input_2d,data_gt_xyz],dim=2)

                noisy_input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
                targets_3d = data_gt_xyz

                # to cuda
                targets_uvxyz, targets_3d, noisy_input_uvxyz = \
                    targets_uvxyz.to(self.device), targets_3d.to(self.device), noisy_input_uvxyz.to(self.device)
     
                n = targets_3d.size(0)
                x = targets_uvxyz
                b = self.betas 

                if random == True:
                    e = torch.randn_like(x)
                    t = torch.randint(low=0, high=self.num_timesteps,
                                  size=(n // 2 + 1,)).to(self.device)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                    # generate x_t (refer to DDIM equation)
                    x = x * a.sqrt() + e * (1.0 - a).sqrt()
                else:
                    mu = (noisy_input_uvxyz-x)/2
                    std = torch.std(noisy_input_uvxyz-x)

                    e = std*torch.randn_like(x) + mu

                    t = torch.randint(low=0, high=self.num_timesteps,
                                  size=(n // 2 + 1,)).to(self.device)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                    # generate x_t (refer to DDIM equation)
                    x = x * a.sqrt() + e * (1.0 - a).sqrt()
                    
                output_noise = self.model_diff(x, src_mask, t.float(), 0)
                loss_diff = (e - output_noise).square().sum(dim=(1, 2)).mean(dim=0)
                """NOISE ESTIMATION LOSS-end-"""
                optimizer.zero_grad()
                loss_diff.backward() 
                
                torch.nn.utils.clip_grad_norm_(
                    self.model_diff.parameters(), config.optim.grad_clip)                
                optimizer.step()
            
                epoch_loss_diff.update(loss_diff.item(), n)

                loss_list.append(epoch_loss_diff.avg)
            
                if self.config.model.ema:
                    ema_helper.update(self.model_diff)
                
                if i%1000 == 0 and i != 0:
                    logging.info('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'\
                        .format(epoch, i+1, len(dataloader), step, data_time, epoch_loss_diff.avg))
                
            data_start = time.time()

            if epoch % decay == 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma) 
                
            if epoch % 1 == 0:
                states = [
                    self.model_diff.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                logging.info('test the performance of current model')

                p1, p2 = self.test_hyber(is_train=True)
                p3, p4 = self.test_hyber_51(is_train=True)
                p5, p6 = self.test_hyber_wo_model(is_train=True)

                p1_list.append(p1)
                p3_list.append(p3)

                print(len(loss_list))
                loss_mean_list.append(np.mean(loss_list))

                loss_list = []

                data = {'DDIM': p1_list, 'DDPM': p3_list, 'LOSS': loss_mean_list}
                # Open the JSON file in write mode
                with open(json_file_path, 'w') as json_file:
                    # Write the data to the JSON file
                    json.dump(data, json_file)

                if p1 < best_p1:
                    best_p1 = p1
                    best_epoch1 = epoch
                
                logging.info('| ################################################################################## |')
                test_timesteps= config.testing.test_timesteps
                
                logging.info('| Best Epoch {:0>4d} STEPS: {:0>4d} MPJPE: {:.2f} | Epoch: {:0>4d} MPJEPE: {:.2f} PA-MPJPE: {:.2f} |'\
                    .format(test_timesteps, best_epoch1, best_p1, epoch, p1, p2))
                
                if p3 < best_p3:
                    torch.save(states,os.path.join(self.args.log_path, "ckpt_{}.pth".format(epoch)))
                    torch.save(states, os.path.join(self.args.log_path, "{}.pth".format(self.args.doc)))
                    print("MODEL SAVED IN: ")
                    print(os.path.join(self.args.log_path, "ckpt_{}.pth".format(epoch)))

                    best_p3 = p3
                    best_epoch2 = epoch
                
                logging.info('| Best Epoch {:0>4d} STEPS: {:0>4d} MPJPE: {:.2f} | Epoch: {:0>4d} MPJEPE: {:.2f} PA-MPJPE: {:.2f} |'\
                    .format(config.diffusion.num_diffusion_timesteps, best_epoch2, best_p3, epoch, p3, p4))

                logging.info('| Without using the model | Epoch: {:0>4d} MPJEPE: {:.2f} PA-MPJPE: {:.2f} |'\
                    .format(epoch, p5, p6))
                logging.info('| ################################################################################## |')
                
    def test_hyber(self, is_train=False):
        ddpm = False #True

        cudnn.benchmark = True
        
        args, config, src_mask = self.args, self.config, self.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample

        dataloader = config.dataloader

        logging.info("Starting the process...")

        print("diffusion.num_diffusion_timesteps: ", config.diffusion.num_diffusion_timesteps)
        print("test_timesteps: ", test_timesteps) 
        print("test_num_diffusion_timesteps", test_num_diffusion_timesteps)
        print("skip: ", test_num_diffusion_timesteps // test_timesteps)
        data_start = time.time()
        data_time = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model_diff.eval()
        self.model_pose.eval()
        
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
            print("seq: ", list(seq))
            logging.info('#steps: {}'.format(len(list(seq))))

        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()

        inference_time = 0
        
        for i, (inputs_xyz, input_2d, targets_3d) in enumerate(dataloader):
            data_start = time.time()

            inputs_xyz, input_2d, targets_3d = inputs_xyz.to(self.device), input_2d.to(self.device), targets_3d.to(self.device)

            input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
            
            input_uvxyz = input_uvxyz.repeat(test_times,1,1)
            # prepare the diffusion parameters
            x = input_uvxyz.clone()
            start = time.time()
            
            if(ddpm == True):
                output_uvxyz = ddpm_steps(x, src_mask, seq, self.model_diff, self.betas, eta=self.args.eta)
            else:
                output_uvxyz = generalized_steps(x, src_mask, seq, self.model_diff, self.betas, eta=self.args.eta)
            
            #output_uvxyz = x
            end = time.time()
            inference_time += end - start
            if i == 0:
                logging.info("Time of inference: {time:.6f}".format(time=inference_time))

            output_uvxyz = output_uvxyz[0][-1]            
            output_uvxyz = torch.mean(output_uvxyz.reshape(test_times,-1,12,5),0)
            
            output_xyz = output_uvxyz[:,:,2:]
            
            output_xyz[:, :, :] = output_xyz[:, :, :] - (output_xyz[:, :1, :] + output_xyz[:,3:4,:])/2
            
            targets_3d[:, :, :] = targets_3d[:, :, :] - (targets_3d[:, :1, :] + targets_3d[:,3:4,:])/2
            
            if ddpm == True:
                output_xyz = output_xyz.to(self.device) # needed when using ddpm 

            epoch_loss_3d_pos.update(mpjpe(output_xyz, targets_3d).item() * 1000.0, targets_3d.size(0))
            epoch_loss_3d_pos_procrustes.update(p_mpjpe(output_xyz.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0, targets_3d.size(0))\
            
            data_time += time.time() - data_start
        
            if i%500 == 0 and i != 0:
                logging.info('({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                        .format(batch=i + 1, size=len(dataloader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                            e2=epoch_loss_3d_pos_procrustes.avg))
                
        logging.info('sum ({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                .format(batch=i + 1, size=len(dataloader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                    e2=epoch_loss_3d_pos_procrustes.avg))
        
        return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg

    # Doing all the steps
    def test_hyber_51(self, is_train=False):
        ddpm = False #True

        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample


        dataloader = config.dataloader

        test_timesteps = config.diffusion.num_diffusion_timesteps
        test_num_diffusion_timesteps = config.diffusion.num_diffusion_timesteps
        logging.info("Starting the process... All the steps")
        print("diffusion.num_diffusion_timesteps: ", config.diffusion.num_diffusion_timesteps)
        print("test_timesteps: ", test_timesteps) 
        print("test_num_diffusion_timesteps", test_num_diffusion_timesteps)
        print("skip: ", test_num_diffusion_timesteps // test_timesteps)
        data_start = time.time()
        data_time = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model_diff.eval()
        self.model_pose.eval()
        
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
            print("seq: ", list(seq))
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()
    
        inference_time = 0
        for i, (inputs_xyz, input_2d, targets_3d) in enumerate(dataloader):
            data_start = time.time()

            inputs_xyz, input_2d, targets_3d = inputs_xyz.to(self.device), input_2d.to(self.device), targets_3d.to(self.device)
            input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
                        
            # generate distribution
            input_uvxyz = input_uvxyz.repeat(test_times,1,1)
            # prepare the diffusion parameters
            x = input_uvxyz.clone()
            start = time.time()
            
            if(ddpm == True):
                output_uvxyz = ddpm_steps(x, src_mask, seq, self.model_diff, self.betas, eta=self.args.eta)
            else:
                output_uvxyz = generalized_steps(x, src_mask, seq, self.model_diff, self.betas, eta=self.args.eta)
            
            #output_uvxyz = x
            end = time.time()
            inference_time += end - start
            if i == 0:
                logging.info("Time of inference: {time:.6f}".format(time=inference_time))

            #Da decommentare
            output_uvxyz = output_uvxyz[0][-1]            
            output_uvxyz = torch.mean(output_uvxyz.reshape(test_times,-1,12,5),0)
            
            output_xyz = output_uvxyz[:,:,2:]

            
            output_xyz[:, :, :] = output_xyz[:, :, :] - (output_xyz[:, :1, :] + output_xyz[:,3:4,:])/2
            
            targets_3d[:, :, :] = targets_3d[:, :, :] - (targets_3d[:, :1, :] + targets_3d[:,3:4,:])/2
            
            if ddpm == True:
                output_xyz = output_xyz.to(self.device) # needed when using ddpm 

            epoch_loss_3d_pos.update(mpjpe(output_xyz, targets_3d).item() * 1000.0, targets_3d.size(0))
            epoch_loss_3d_pos_procrustes.update(p_mpjpe(output_xyz.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0, targets_3d.size(0))\
            
            data_time += time.time() - data_start
        
            if i%500 == 0 and i != 0:
                logging.info('({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                        .format(batch=i + 1, size=len(dataloader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                            e2=epoch_loss_3d_pos_procrustes.avg))
                
        logging.info('sum ({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                .format(batch=i + 1, size=len(dataloader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                    e2=epoch_loss_3d_pos_procrustes.avg))

        return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    parser.add_argument("--seed", type=int, default=19960903, help="Random seed")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the config file")
    parser.add_argument("--exp", type=str, default="exp", 
                        help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, required=True, 
                        help="A string for documentation purpose. "\
                            "Will be the name of the log folder.", )
    parser.add_argument("--verbose", type=str, default="info", 
                        help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--ni", action="store_true",
                        help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    ### Difformer configuration ####
    # Diffusion process hyperparameters
    parser.add_argument("--skip_type", type=str, default="uniform",
                        help="skip according to (uniform or quad(quadratic))")
    parser.add_argument("--eta", type=float, default=0.0, 
                        help="eta used to control the variances of sigma")
    parser.add_argument("--sequence", action="store_true")
    # Diffusion model parameters
    parser.add_argument('--n_head', type=int, default=4, help='num head')
    parser.add_argument('--dim_model', type=int, default=96, help='dim model')
    parser.add_argument('--n_layer', type=int, default=5, help='num layer')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                        help='downsample frame rate by factor')
    # load pretrained model
    parser.add_argument('--model_diff_path', default=None, type=str,
                        help='the path of pretrain model')
    parser.add_argument('--model_pose_path', default=None, type=str,
                        help='the path of pretrain model')
    parser.add_argument('--train', action = 'store_true',
                        help='train or evluate')
    #training hyperparameter
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--lr_gamma', default=0.9, type=float, metavar='N',
                        help='weight decay rate')
    parser.add_argument('--lr', default=1e-5, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('--decay', default=60, type=int, metavar='N',
                        help='decay frequency(epoch)')
    #test hyperparameter
    parser.add_argument('--test_times', default=5, type=int, metavar='N',
                    help='the number of test times')
    parser.add_argument('--test_timesteps', default=50, type=int, metavar='N',
                    help='the number of test time steps')
    parser.add_argument('--test_num_diffusion_timesteps', default=500, type=int, metavar='N',
                    help='the number of test times')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device
    # update configure file
    new_config.training.batch_size = args.batch_size
    new_config.optim.lr = args.lr
    new_config.optim.lr_gamma = args.lr_gamma
    new_config.optim.decay = args.decay

    new_config.folder_path = new_config.data.dataset_path_train_3d #"data/gaussian16"
    new_config.folder_path_gt = new_config.data.dataset_path_gt #"data/gt"

    print(new_config.folder_path)

    print(new_config.folder_path_gt)
    logging.info('==> Using settings {}'.format(args))
    logging.info('==> Using configures {}'.format(new_config))

    if args.train:
        if os.path.exists(args.log_path):
            overwrite = False
            if args.ni:
                overwrite = True
            else:
                response = input("Folder already exists. Overwrite? (Y/N)")
                if response.upper() == "Y":
                    overwrite = True

            if overwrite:
                shutil.rmtree(args.log_path)
                os.makedirs(args.log_path)
            else:
                print("Folder exists. Program halted.")
                sys.exit(0)
        else:
            os.makedirs(args.log_path)

        with open(os.path.join(args.log_path, "config.yml"), "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    
    try:
        runner = Diffpose(args, config)
        runner.create_diffusion_model(args.model_diff_path)
        print("Start process data")
        p00, p01 = runner.prepare_data_from_csv()
        
        print("Finish process data")
        if args.train:
            runner.train()
        else:
            p1, p2 = runner.test_hyber()
            p3, p4 = runner.test_hyber_51()
            print("#############################################")
            head, folder = os.path.split(config.folder_path)
            
            print("Dataset test: ", folder)
            print("Without the model")
            print('| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                        .format(e1=p00, e2=p01))
            print("Using the model with 2 steps")
            print('| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                        .format(e1=p1, e2=p2))
            print("Using the model with all the steps")
            print('| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                        .format(e1=p3, e2=p4))
            print("#############################################")
    except Exception:
        logging.error(traceback.format_exc())

    return 0

if __name__ == "__main__":
    sys.exit(main())
