import logging
import os
import torch

torch.set_printoptions(sci_mode=False)

import os
import logging
import time

import torch

import pandas as pd
#from torch.nn.parallel import DistributedDataParallel 

from models.gcndiff import GCNdiff, adj_mx_from_edges

from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps, ddpm_steps
from common.loss import mpjpe, p_mpjpe

from torch.utils.data import TensorDataset, DataLoader

def evaluate_seq(self, input_xyz, x, lut):
        output_uvxyz = generalized_steps(x, self.src_mask, [50], self.model_diff, self.betas, eta=self.args.eta)

        output_uvxyz = output_uvxyz[0][-1]            
        output_uvxyz = torch.mean(output_uvxyz.reshape(self.config.testing.test_times,-1,12,5),0)
        
        output_xyz = output_uvxyz[:,:,2:]

        output_xyz[:, :, :] = output_xyz[:, :, :] - (output_xyz[:, :1, :] + output_xyz[:,3:4,:])/2

        mpjpe_inout = mpjpe(output_xyz, input_xyz).item() * 1000.0
        minimum = lut[lut['start_bin'] < mpjpe_inout]
        row = minimum[minimum['end_bin'] > mpjpe_inout]
        step = row['step']

        return [step.values[0]], mpjpe_inout

def run(self, treshold = 150, smart = False, n_step =0,  mode = "LR"):
    lut = pd.read_csv('checkpoints/lut.csv')
    print(mode)
    
    args, config, src_mask = self.args, self.config, self.src_mask
    test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
        config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
    dataloader = config.dataloader

    logging.info("Starting the process...")
    
    data_start = time.time()
    data_time = 0

    # Switch to test mode
    torch.set_grad_enabled(False)
    self.model_diff.eval()

    inference_time = 0
    
    output_xyz_list = []
    mpjpe_list = []
    mpjpe_list_noisy_denoised = []      
    
    step_list = []

    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()
    
    in_mpjpe = []
    mpjpe_in_gt, mpjpe_in_out50, step_list = [], [], []
    print(len(dataloader))
    for i, (inputs_xyz, input_2d, targets_3d) in enumerate(dataloader):
        data_start = time.time()

        inputs_xyz, input_2d, targets_3d = inputs_xyz.to(self.device), input_2d.to(self.device), targets_3d.to(self.device)

        input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
        
        input_uvxyz = input_uvxyz.repeat(test_times,1,1)
        #input_uvxyz=torch.nan_to_num(input_uvxyz, nan=0.0)
        #print(input_uvxyz.isnan())
        input_uvxyz[input_uvxyz.isnan()] = 0.0
        # prepare the diffusion parameters
        x = input_uvxyz.clone()

        start = time.time()

        seq, mpjpe_inout50=  evaluate_seq(self,inputs_xyz, x,lut)

        step_list.append(seq[0])

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

        epoch_loss_3d_pos.update(mpjpe(output_xyz, targets_3d).item() * 1000.0, targets_3d.size(0))
        #epoch_loss_3d_pos_procrustes.update(p_mpjpe(output_xyz.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0, targets_3d.size(0))\
        in_mpjpe.append(mpjpe(inputs_xyz, targets_3d).item() * 1000.0)
        
        data_time += time.time() - data_start

        output_xyz_list.append(output_xyz)
        mpjpe_list.append(mpjpe(output_xyz, targets_3d).item() * 1000.0)
        mpjpe_list_noisy_denoised.append(mpjpe(output_xyz, inputs_xyz).item() * 1000.0)

        mpjpe_in_out50.append(mpjpe_inout50)

        mpjpe_in_gt.append(mpjpe(inputs_xyz,targets_3d).item()*1000)
        
        if i % 1000 == 0:
            out_table = pd.DataFrame({'mpjpe_in_gt':mpjpe_in_gt, 'mpjpe_in_out50':mpjpe_in_out50, 'mpjpe_out_gt':mpjpe_list, 'step_list': step_list})
            out_table.to_csv('lut_rt_gaussian_s11s9.csv')
           
        if i%500 == 0 and i != 0:
            logging.info('({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                    .format(batch=i + 1, size=len(dataloader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                        e2=epoch_loss_3d_pos_procrustes.avg))

    out_table = pd.DataFrame({'mpjpe_in_gt':mpjpe_in_gt, 'mpjpe_in_out50':mpjpe_in_out50, 'mpjpe_out_gt':mpjpe_list, 'step_list': step_list})

    return out_table

import datetime

def run_and_save(self):
    lut = pd.read_csv('checkpoints/lut.csv')
    
    args, config, src_mask = self.args, self.config, self.src_mask
    test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
        config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
    dataloader = config.dataloader

    logging.info("Starting the process...")
    
    data_start = time.time()
    data_time = 0

    # Switch to test mode
    torch.set_grad_enabled(False)
    self.model_diff.eval()

    inference_time = 0
    
    output_xyz_list = []
    mpjpe_list = []
    mpjpe_list_noisy_denoised = []      
    
    step_list = []

    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()
    
    in_mpjpe = []
    mpjpe_in_gt, mpjpe_in_out50, step_list = [], [], []
    print(len(dataloader))
    for i, (inputs_xyz, input_2d, targets_3d) in enumerate(dataloader):
        data_start = time.time()

        inputs_xyz, input_2d, targets_3d = inputs_xyz.to(self.device), input_2d.to(self.device), targets_3d.to(self.device)

        input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
        
        input_uvxyz = input_uvxyz.repeat(test_times,1,1)
        #input_uvxyz=torch.nan_to_num(input_uvxyz, nan=0.0)
        #print(input_uvxyz.isnan())
        input_uvxyz[input_uvxyz.isnan()] = 0.0
        # prepare the diffusion parameters
        x = input_uvxyz.clone()

        start = time.time()

        seq, mpjpe_inout50=  evaluate_seq(self,inputs_xyz, x,lut)

        step_list.append(seq[0])

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

        epoch_loss_3d_pos.update(mpjpe(output_xyz, targets_3d).item() * 1000.0, targets_3d.size(0))
        #epoch_loss_3d_pos_procrustes.update(p_mpjpe(output_xyz.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0, targets_3d.size(0))\
        in_mpjpe.append(mpjpe(inputs_xyz, targets_3d).item() * 1000.0)
        
        data_time += time.time() - data_start

        output_xyz_list.append(output_xyz)
        mpjpe_list.append(mpjpe(output_xyz, targets_3d).item() * 1000.0)
        mpjpe_list_noisy_denoised.append(mpjpe(output_xyz, inputs_xyz).item() * 1000.0)

        mpjpe_in_out50.append(mpjpe_inout50)

        mpjpe_in_gt.append(mpjpe(inputs_xyz,targets_3d).item()*1000)
        
        if i % 1000 == 0:
            current_time = datetime.datetime.now()
            print(str(i) + " / " + str(len(dataloader)) + " " + "\t" + str(current_time))      
                
            out_table = pd.DataFrame({'mpjpe_in_gt':mpjpe_in_gt, 'mpjpe_in_out50':mpjpe_in_out50, 'mpjpe_out_gt':mpjpe_list, 'step_list': step_list})
            out_table.to_csv(config.file_csv_name)

    out_table = pd.DataFrame({'mpjpe_in_gt':mpjpe_in_gt, 'mpjpe_in_out50':mpjpe_in_out50, 'mpjpe_out_gt':mpjpe_list, 'step_list': step_list})
    out_table.to_csv(config.file_csv_name)

    return out_table

def run_and_save_total_cap(self):
    lut = pd.read_csv('checkpoints/lut.csv')
    
    args, config, src_mask = self.args, self.config, self.src_mask
    test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
        config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
    dataloader = config.dataloader

    logging.info("Starting the process...")
    
    data_start = time.time()
    data_time = 0

    # Switch to test mode
    torch.set_grad_enabled(False)
    self.model_diff.eval()

    inference_time = 0
    
    output_xyz_list = []
    mpjpe_list = []
    mpjpe_list_noisy_denoised = []      
    
    step_list = []

    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()
    
    name_list = []
    in_mpjpe = []
    mpjpe_in_gt, mpjpe_in_out50, step_list = [], [], []
    print(len(dataloader))
    for i, (inputs_xyz, input_2d, targets_3d, name) in enumerate(dataloader):
        data_start = time.time()

        inputs_xyz, input_2d, targets_3d = inputs_xyz.to(self.device), input_2d.to(self.device), targets_3d.to(self.device)

        input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
        
        input_uvxyz = input_uvxyz.repeat(test_times,1,1)
        input_uvxyz[input_uvxyz.isnan()] = 0.0
        # prepare the diffusion parameters
        x = input_uvxyz.clone()

        start = time.time()

        seq, mpjpe_inout50=  evaluate_seq(self,inputs_xyz, x,lut)

        step_list.append(seq[0])

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

        epoch_loss_3d_pos.update(mpjpe(output_xyz, targets_3d).item() * 1000.0, targets_3d.size(0))
        #epoch_loss_3d_pos_procrustes.update(p_mpjpe(output_xyz.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0, targets_3d.size(0))\
        in_mpjpe.append(mpjpe(inputs_xyz, targets_3d).item() * 1000.0)
        
        data_time += time.time() - data_start

        output_xyz_list.append(output_xyz)
        mpjpe_list.append(mpjpe(output_xyz, targets_3d).item() * 1000.0)
        mpjpe_list_noisy_denoised.append(mpjpe(output_xyz, inputs_xyz).item() * 1000.0)

        mpjpe_in_out50.append(mpjpe_inout50)

        mpjpe_in_gt.append(mpjpe(inputs_xyz,targets_3d).item()*1000)

        string = next(key for key, value in config.mapping.items() if value == name)
            
        name_list.append(string)

        if i % 1000 == 0:
            current_time = datetime.datetime.now()
            print(str(i) + " / " + str(len(dataloader)) + " " +  "\t" + str(current_time))      
                
            out_table = pd.DataFrame({'mpjpe_in_gt':mpjpe_in_gt, 'mpjpe_in_out50':mpjpe_in_out50, 'mpjpe_out_gt':mpjpe_list, 'step_list': step_list, 'name':name_list})
            out_table.to_csv(config.file_csv_name)

    out_table = pd.DataFrame({'mpjpe_in_gt':mpjpe_in_gt, 'mpjpe_in_out50':mpjpe_in_out50, 'mpjpe_out_gt':mpjpe_list, 'step_list': step_list, 'name':name_list})
    out_table.to_csv(config.file_csv_name)

    return out_table


def read_noisy_and_gt_csv_files(self, folder_path, folder_path_gt, csv_file):
            
            concatenated_df = pd.DataFrame()
            concatenated_df_gt = pd.DataFrame()

            csv_file_path = os.path.join(folder_path, csv_file)
            
            csv_file_path_gt = os.path.join(folder_path_gt, csv_file)

            dataframe = pd.read_csv(csv_file_path)
            dataframe_gt = pd.read_csv(csv_file_path_gt)

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

            return tensor_from_csv_uv, tensor_from_csv_xyz, tensor_from_csv_uv_gt, tensor_from_csv_xyz_gt


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a

def generalized_steps(x, src_mask, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            if torch.cuda.is_available():
                t = (torch.ones(n) * i).cuda()
                next_t = (torch.ones(n) * j).cuda()
            else:
                t = (torch.ones(n) * i)
                next_t = (torch.ones(n) * j)
                
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1]
            et = model(xt, src_mask, t.float(), 0)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)

    return xs, x0_preds

def create_diffusion_model(self, model_path = None):
        args, config = self.args, self.config
        # EDGES 12
        edges = torch.tensor([[0, 1], [0, 9],[1, 2], [0, 3], [3, 6], [3, 4], [4, 5],
                            [6, 7], [6, 9], [7, 8], [9, 10], [10, 11]], dtype=torch.long)
        
        adj = adj_mx_from_edges(num_pts=12, edges=edges, sparse=False)
        self.model_diff = GCNdiff(adj.cuda(), config).cuda()
        
        self.model_diff = torch.nn.DataParallel(self.model_diff, device_ids=[0])
        
        # load pretrained model
        if model_path:
            states = torch.load(model_path)
            self.model_diff.load_state_dict(states[0])
