import logging
import torch
import numpy as np

torch.set_printoptions(sci_mode=False)

import logging
import time

import numpy as np
import torch
import pandas as pd

from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps, ddpm_steps
from common.loss import mpjpe, p_mpjpe

import datetime

class RealTimeDiffusion:

    def __init__(self, runner, name_csv):
        self.name_csv = name_csv
        self.runner = runner
        self.slopes = []
        self.hist=[]
        self.list_len = 0

    def evaluate_seq(self, input_xyz, x, lut):
        output_uvxyz = generalized_steps(x, self.runner.src_mask, [50], self.runner.model_diff, self.runner.betas, eta=self.runner.args.eta)

        output_uvxyz = output_uvxyz[0][-1]            
        output_uvxyz = torch.mean(output_uvxyz.reshape(self.runner.config.testing.test_times,-1,12,5),0)
        
        output_xyz = output_uvxyz[:,:,2:]

        output_xyz[:, :, :] = output_xyz[:, :, :] - (output_xyz[:, :1, :] + output_xyz[:,3:4,:])/2

        mpjpe_inout = mpjpe(output_xyz, input_xyz).item() * 1000.0

        self.hist.append(mpjpe_inout)

        minimum = lut[lut['start_bin'] < mpjpe_inout]
        row = minimum[minimum['end_bin'] > mpjpe_inout]
        maxstep = int(row['step'].values)
        
        if len(self.hist) < 2:
            self.list_len = 5
            return 1, maxstep, [maxstep]
        else:
            m = self.hist[-1] - self.hist[-2]

            self.hist=self.hist[-1:]

            if m>0:
                self.list_len+=1
            elif m < 0:
                self.list_len-=1

            if self.list_len > 5:
                self.list_len = 5
            if self.list_len < 1:
                self.list_len = 1

            if self.list_len == 1:
                return self.list_len, maxstep, [maxstep]
            
            return self.list_len, maxstep, list(map(int, np.round(np.linspace(0, maxstep, self.list_len))))

    def run(self, treshold = 150, smart = False, n_step =0,  mode = "LR"):
        print(mode)
        lut = pd.read_csv('checkpoints/lut.csv')
        
        args, config, src_mask = self.runner.args, self.runner.config, self.runner.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
        dataloader = config.dataloader

        logging.info("Starting the process...")
        
        data_start = time.time()
        data_time = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.runner.model_diff.eval()
        
        inference_time = 0
        
        output_xyz_list = []
        mpjpe_list = []
        mpjpe_list_noisy_denoised = []
        
        len_list_list = []

        in_mpjpe = []
        for i, (inputs_xyz, input_2d, targets_3d) in enumerate(dataloader):
    
            data_start = time.time()

            inputs_xyz, input_2d, targets_3d = inputs_xyz.to(self.runner.device), input_2d.to(self.runner.device), targets_3d.to(self.runner.device)

            input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
            
            input_uvxyz = input_uvxyz.repeat(test_times,1,1)
            input_uvxyz[input_uvxyz.isnan()] = 0.0

            # prepare the diffusion parameters
            x = input_uvxyz.clone()
            len_list, seq = self.evaluate_seq(inputs_xyz, x, lut)

            start = time.time()

            output_uvxyz = generalized_steps(x, src_mask, seq, self.runner.model_diff, self.runner.betas, eta=self.runner.args.eta)
            
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
    
            in_mpjpe.append(mpjpe(inputs_xyz, targets_3d).item() * 1000.0)
            
            data_time += time.time() - data_start

            output_xyz_list.append(output_xyz)
            mpjpe_list.append(mpjpe(output_xyz, targets_3d).item() * 1000.0)
            mpjpe_list_noisy_denoised.append(mpjpe(output_xyz, inputs_xyz).item() * 1000.0)

            self.hist.append(mpjpe(output_xyz, inputs_xyz).item() * 1000.0)

            len_list_list.append(len_list)

        return output_xyz_list, mpjpe_list, mpjpe_list_noisy_denoised, len_list_list, in_mpjpe, self.slopes

    def run_and_save(self):
        lut = pd.read_csv('checkpoints/lut.csv')
        
        args, config, src_mask = self.runner.args, self.runner.config, self.runner.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
        dataloader = config.dataloader

        logging.info("Starting the process...")
        
        data_start = time.time()
        data_time = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.runner.model_diff.eval()
        
        inference_time = 0
        
        output_xyz_list = []
        mpjpe_list = []
        mpjpe_list_noisy_denoised = []
        
        len_list_list = []
        maxstep_list = []
        in_mpjpe = []
        for i, (inputs_xyz, input_2d, targets_3d) in enumerate(dataloader):
            #print(seq) 
            data_start = time.time()

            inputs_xyz, input_2d, targets_3d = inputs_xyz.to(self.runner.device), input_2d.to(self.runner.device), targets_3d.to(self.runner.device)

            input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
            
            input_uvxyz = input_uvxyz.repeat(test_times,1,1)
            input_uvxyz[input_uvxyz.isnan()] = 0.0

            # prepare the diffusion parameters
            x = input_uvxyz.clone()
            
            len_list, maxstep, seq = self.evaluate_seq(inputs_xyz, x, lut)

            #print(len_list, seq)
            start = time.time()

            output_uvxyz = generalized_steps(x, src_mask, seq, self.runner.model_diff, self.runner.betas, eta=self.runner.args.eta)
            
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
    
            in_mpjpe.append(mpjpe(inputs_xyz, targets_3d).item() * 1000.0)
            
            data_time += time.time() - data_start

            maxstep_list.append(maxstep)

            output_xyz_list.append(output_xyz)
            mpjpe_list.append(mpjpe(output_xyz, targets_3d).item() * 1000.0)
            mpjpe_list_noisy_denoised.append(mpjpe(output_xyz, inputs_xyz).item() * 1000.0)
            len_list_list.append(len_list)
            
            if i % 1000 == 0:
                current_time = datetime.datetime.now()
                print(str(i) + " / " + str(len(dataloader)) + " " + self.name_csv +  "\t" + str(current_time))                  

                out_table = pd.DataFrame({'mpjpe_in_gt':in_mpjpe, 'mpjpe_in_out50':mpjpe_list_noisy_denoised, 'mpjpe_out_gt':mpjpe_list, '#step': len_list_list, "maxstep": maxstep_list})
                out_table.to_csv(self.name_csv)
            
        out_table = pd.DataFrame({'mpjpe_in_gt':in_mpjpe, 'mpjpe_in_out50':mpjpe_list_noisy_denoised, 'mpjpe_out_gt':mpjpe_list, '#step': len_list_list, "maxstep": maxstep_list})
        out_table.to_csv(self.name_csv)
            
        return output_xyz_list, mpjpe_list, mpjpe_list_noisy_denoised, len_list_list, in_mpjpe, self.slopes, out_table


    def run_and_save_total_cap(self):
        lut = pd.read_csv('checkpoints/lut.csv')
        
        args, config, src_mask = self.runner.args, self.runner.config, self.runner.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
        dataloader = config.dataloader

        logging.info("Starting the process...")
        
        data_start = time.time()
        data_time = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.runner.model_diff.eval()
        
        inference_time = 0
        
        output_xyz_list = []
        mpjpe_list = []
        mpjpe_list_noisy_denoised = []
        
        len_list_list = []
        
        name_list = []
        maxstep_list = []
        in_mpjpe = []
        for i, (inputs_xyz, input_2d, targets_3d, name) in enumerate(dataloader):
            #print(seq) 
            data_start = time.time()

            inputs_xyz, input_2d, targets_3d = inputs_xyz.to(self.runner.device), input_2d.to(self.runner.device), targets_3d.to(self.runner.device)

            input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
            
            input_uvxyz = input_uvxyz.repeat(test_times,1,1)
            input_uvxyz[input_uvxyz.isnan()] = 0.0

            # prepare the diffusion parameters
            x = input_uvxyz.clone()
            
            len_list, maxstep, seq = self.evaluate_seq(inputs_xyz, x, lut)

            #print(len_list, seq)
            start = time.time()

            output_uvxyz = generalized_steps(x, src_mask, seq, self.runner.model_diff, self.runner.betas, eta=self.runner.args.eta)
            
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
    
            in_mpjpe.append(mpjpe(inputs_xyz, targets_3d).item() * 1000.0)
            
            data_time += time.time() - data_start

            maxstep_list.append(maxstep)

            output_xyz_list.append(output_xyz)
            mpjpe_list.append(mpjpe(output_xyz, targets_3d).item() * 1000.0)
            mpjpe_list_noisy_denoised.append(mpjpe(output_xyz, inputs_xyz).item() * 1000.0)
            len_list_list.append(len_list)
            
            string = next(key for key, value in config.mapping.items() if value == name)
            
            name_list.append(string)
            if i % 1000 == 0:
                current_time = datetime.datetime.now()
                print(str(i) + " / " + str(len(dataloader)) + " " + self.name_csv +  "\t" + str(current_time))                  

                out_table = pd.DataFrame({'mpjpe_in_gt':in_mpjpe, 'mpjpe_in_out50':mpjpe_list_noisy_denoised, 'mpjpe_out_gt':mpjpe_list, '#step': len_list_list, "maxstep": maxstep_list, 'name':name_list})
                out_table.to_csv(self.name_csv)
    
        out_table = pd.DataFrame({'mpjpe_in_gt':in_mpjpe, 'mpjpe_in_out50':mpjpe_list_noisy_denoised, 'mpjpe_out_gt':mpjpe_list, '#step': len_list_list, "maxstep": maxstep_list, 'name':name_list})
        out_table.to_csv(self.name_csv)
            
        return output_xyz_list, mpjpe_list, mpjpe_list_noisy_denoised, len_list_list, in_mpjpe, self.slopes, out_table
