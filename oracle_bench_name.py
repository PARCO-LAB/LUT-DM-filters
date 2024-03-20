import logging
import os
import torch
import numpy as np
import argparse

torch.set_printoptions(sci_mode=False)

import datetime
import os
import logging
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import pandas as pd
#from torch.nn.parallel import DistributedDataParallel 

from models.gcndiff import GCNdiff, adj_mx_from_edges

from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps, ddpm_steps
from common.loss import mpjpe, p_mpjpe

from torch.utils.data import TensorDataset, DataLoader

def seed_torch(seed=42):
    # Set seed for CPU
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set seed for GPU if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

def test_hyber_seq(self):
        cudnn.benchmark = True
        
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

        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()

        inference_time = 0
        
        input_keypoints_list = []
        gt_keypoints_list = []
        input_2d_keypoints_list = []

        mpjpe_in_gt, mpjpe_in_out50, mpjpe_out50_gt, best_step, mpjpe_outbeststep_gt, nomi = [], [], [], [], [], []
        for i, (inputs_xyz, input_2d, targets_3d) in enumerate(dataloader):

            data_start = time.time()
           
            inputs_xyz, input_2d, targets_3d = inputs_xyz.to(self.device), input_2d.to(self.device), targets_3d.to(self.device)

            input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
            
            input_uvxyz = input_uvxyz.repeat(test_times,1,1)

            # prepare the diffusion parameters
            x = input_uvxyz.clone()
            
            mpjpes = []
            for step in range(51):
                seq = [step]
                start = time.time()

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
                #output_xyz[:, :, :] -= output_xyz[:, :1, :]

                targets_3d[:, :, :] = targets_3d[:, :, :] - (targets_3d[:, :1, :] + targets_3d[:,3:4,:])/2

                epoch_loss_3d_pos.update(mpjpe(output_xyz, targets_3d).item() * 1000.0, targets_3d.size(0))

                data_time += time.time() - data_start

                #output_xyz_list.append(output_xyz)
                #mpjpe_list.append(mpjpe(output_xyz, targets_3d).item() * 1000.0)
                #mpjpe_list_noisy_denoised.append(mpjpe(output_xyz, inputs_xyz).item() * 1000.0)
                mpjpes.append(mpjpe(output_xyz, targets_3d).item() * 1000.0)
                if step == 50:
                    mpjpe_in_out50.append(mpjpe(inputs_xyz,output_xyz).item()*1000)
                    mpjpe_out50_gt.append(mpjpe(output_xyz,targets_3d).item()*1000)
                    input_2d_keypoints_list.append(input_2d.cpu().view(-1).numpy())
                    input_keypoints_list.append(inputs_xyz.cpu().view(-1).numpy())
                    gt_keypoints_list.append(targets_3d.cpu().view(-1).numpy())
            mpjpe_outbeststep_gt.append(np.min(mpjpes))
            best_step.append(np.argmin(mpjpes))
            mpjpe_in_gt.append(mpjpe(inputs_xyz,targets_3d).item()*1000)
            #nomi.append(nome.item())

            #if i == 300:
            #    break
            if i % 100 == 0:
                current_time = datetime.datetime.now()
                print(str(i) + " / " + str(len(dataloader)) + " " + "\t" + str(current_time))      
                print("Saved in: " +  config.file_csv_name)            

                column_names = [f'in2d_{i}' for i in range(len(input_2d_keypoints_list[0]))]
                df = pd.DataFrame(input_2d_keypoints_list, columns=column_names)
                
                column_names = [f'in_{i}' for i in range(len(input_keypoints_list[0]))]
                df1 = pd.DataFrame(input_keypoints_list, columns=column_names)

                column_names = [f'gt_{i}' for i in range(len(gt_keypoints_list[0]))]
                df2 = pd.DataFrame(gt_keypoints_list, columns=column_names)
                
                out_table = pd.DataFrame({'mpjpe_in_gt':mpjpe_in_gt, 'mpjpe_in_out50':mpjpe_in_out50, 'mpjpe_out50_gt':mpjpe_out50_gt, 'best_step':best_step, 'mpjpe_outbeststep_gt':mpjpe_outbeststep_gt})

                combined_df = pd.concat([out_table, df, df1, df2], axis=1)

                combined_df.to_csv( config.file_csv_name) 
                

            if i%500 == 0 and i != 0:
                logging.info('({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                        .format(batch=i + 1, size=len(dataloader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                            e2=epoch_loss_3d_pos_procrustes.avg))

        column_names = [f'in2d_{i}' for i in range(len(input_2d_keypoints_list[0]))]
        df = pd.DataFrame(input_2d_keypoints_list, columns=column_names)
        
        column_names = [f'in_{i}' for i in range(len(input_keypoints_list[0]))]
        df1 = pd.DataFrame(input_keypoints_list, columns=column_names)

        column_names = [f'gt_{i}' for i in range(len(gt_keypoints_list[0]))]
        df2 = pd.DataFrame(gt_keypoints_list, columns=column_names)
        
        out_table = pd.DataFrame({'mpjpe_in_gt':mpjpe_in_gt, 'mpjpe_in_out50':mpjpe_in_out50, 'mpjpe_out50_gt':mpjpe_out50_gt, 'best_step':best_step, 'mpjpe_outbeststep_gt':mpjpe_outbeststep_gt})

        combined_df = pd.concat([out_table, df, df1, df2], axis=1)

        combined_df.to_csv( config.file_csv_name)       
        return combined_df

def read_noisy_and_gt_csv_files(self, folder_path, folder_path_gt, csv_file):
    
    concatenated_df = pd.DataFrame()
    concatenated_df_gt = pd.DataFrame()

    csv_file_path = os.path.join(folder_path, csv_file)
    
    csv_file_path_gt = os.path.join(folder_path_gt, csv_file)

    dataframe = pd.read_csv(csv_file_path)
    dataframe_gt = pd.read_csv(csv_file_path_gt)

    concatenated_df = pd.concat([concatenated_df, dataframe], axis=0, ignore_index=True)
    concatenated_df_gt = pd.concat([concatenated_df_gt, dataframe_gt], axis=0, ignore_index=True)

    concatenated_df /= 1000
    concatenated_df_gt /= 1000

    tensor_from_csv_xyz = self.data_from_dataframe_12_xyz(concatenated_df)
    tensor_from_csv_uv = self.data_from_dataframe_12_uv(concatenated_df)

    tensor_from_csv_xyz_gt = self.data_from_dataframe_12_xyz(concatenated_df_gt)
    tensor_from_csv_uv_gt = self.data_from_dataframe_12_uv(concatenated_df_gt)

    return tensor_from_csv_uv, tensor_from_csv_xyz, tensor_from_csv_uv_gt, tensor_from_csv_xyz_gt

def parse_args_and_config(config):
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--path", type=str, required=True, help="Path of the noisy data")
    parser.add_argument("--gt_path", type=str, required=True, help="Path for the ground truth")
    
    args = parser.parse_args()

    new_config = config
    new_config.folder_path = args.path
    new_config.folder_path_gt = args.gt_path

    return new_config

def configs(runner, config):
    config.file_csv_name = 'results/' +  "ORACLE_" +'seq.csv'        
    
    return config


def main():
    from types import SimpleNamespace
    device = SimpleNamespace(type='cuda')
    # Create args namespace
    args = SimpleNamespace(
        batch_size=32,
        decay=60,
        dim_model=96,
        downsample=1,
        dropout=0.25,
        eta=0.0,
        lr=1e-05,
        lr_gamma=0.9,
        model_diff_path='checkpoints/diffusion_model.pth',
        n_head=4,
        n_layer=5,
        ni=True,
        seed=19960903,
        sequence=False,
        skip_type='uniform',
        test_num_diffusion_timesteps=500,
        test_times=5,
        test_timesteps=50,
        train=False,
        verbose='info'
    )

    # Create config namespace
    old_config = SimpleNamespace(
        data=SimpleNamespace(
            dataset='human36m',
            num_joints=17,
            num_workers=32
        ),
        device=device,  
        diffusion=SimpleNamespace(
            beta_end=0.001,
            beta_schedule='linear',
            beta_start=0.0001,
            num_diffusion_timesteps=51
        ),
        model=SimpleNamespace(
            coords_dim=[5, 5],
            dropout=0.25,
            ema=True,
            ema_rate=0.999,
            emd_dim=96,
            hid_dim=96,
            n_head=4,
            n_pts=12,
            num_layer=5,
            resamp_with_conv=True,
            var_type='fixedsmall'
        ),
        optim=SimpleNamespace(
            amsgrad=False,
            decay=60,
            eps=1e-08,
            grad_clip=1.0,
            lr=1e-05,
            lr_gamma=0.9,
            optimizer='Adam'
        ),
        testing=SimpleNamespace(
            test_num_diffusion_timesteps=12,
            test_times=1,
            test_timesteps=2
        ),
        folder_path_gt = "",
        training=SimpleNamespace(
            batch_size=32,
            n_epochs=80,
            n_iters=5000000,
            num_workers=32,
            random=False,
            snapshot_freq=5000,
            validation_freq=2000
        )
    )

    config = parse_args_and_config(old_config)

    from diffpose_train import Diffpose
    runner = Diffpose(args, config)
    create_diffusion_model(runner, args.model_diff_path)

    config = configs(runner, config)
    print("Start process data")
    
    test_input_2d, test_inputs_xyz, test_input_2d_gt, test_data_gt_xyz, dataset_name = runner.read_noisy_and_gt_csv_files_from_folder_trt(config.folder_path, config.folder_path_gt)
        
    test_inputs_xyz[:, :, :] = test_inputs_xyz[:, :, :] - (test_inputs_xyz[:, :1, :] + test_inputs_xyz[:,3:4,:])/2
    test_data_gt_xyz[:, :, :] = test_data_gt_xyz[:, :, :] - (test_data_gt_xyz[:, :1, :] + test_data_gt_xyz[:,3:4,:])/2
    test_input_2d[:, :, :] = test_input_2d[:, :, :] - (test_input_2d[:, :1, :] + test_input_2d[:,3:4,:])/2

    test_dataset = TensorDataset(test_inputs_xyz, test_input_2d, test_data_gt_xyz)

    batch_size = 1
    config.dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
    
    print("Taking the files from: " + config.folder_path)
    print("Taking the gt files from: " + config.folder_path_gt)
    print("Saved in: " + config.file_csv_name)            


    test_dataset = TensorDataset(test_inputs_xyz, test_input_2d, test_data_gt_xyz)

    batch_size = 1
    config.dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  

    data_out = test_hyber_seq(runner)
    data_out.to_csv(config.file_csv_name)


if __name__ == "__main__":
    main()
