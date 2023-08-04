from __future__ import print_function, division
import os

import torch
import torch.nn.functional as F
import torchio as tio
from monai.data import list_data_collate

from monai.inferers import sliding_window_inference
from utils import net_builder, get_test_dataset, standard_transform, prepare_batch, compute_metrics
import numpy as np
import nibabel as nib
import pathlib as plb
import cc3d
import csv
import sys


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='./saved_models/CC_200_0.0.0/')
    parser.add_argument('--use_train_model', action='store_true')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--donot_use_bn', action = 'store_true', help= 'To kill batch normalisation.')

    '''
    Data Configurations
    '''
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='./data/FDG-PET-CT-Lesions_nifti/')
    parser.add_argument('--num_classes', type=int, default=2)
    
    '''
    GPU
    '''
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.load_path, 'model_best.pth')
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['train_model'] if args.use_train_model else checkpoint['eval_model']
    
    _net_builder = net_builder
    net = _net_builder(num_classes=args.num_classes)
    net.load_state_dict(load_model)
    
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    
    test_set = get_test_dataset(args.data_dir, standard_transform)
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=0, collate_fn = list_data_collate)
 
 
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            roi_size = (128, 128, 32)
            sw_batch_size = 20
            
            X, _ = prepare_batch(data, args.gpu)
            
            mask_out = sliding_window_inference(X, roi_size, sw_batch_size, net)
            mask_out = torch.argmax(mask_out, dim=1).detach().cpu().numpy().squeeze()
            mask_out = mask_out.astype(np.uint8)               
            print("done inference")
            
            dice_sc, false_pos_vol, false_neg_vol = compute_metrics(mask_out, data['segmentation'][tio.DATA][0,1])
            
            csv_header = ['dice_sc', 'false_pos_vol', 'false_neg_vol']
            csv_rows = [[dice_sc], [false_pos_vol], [false_neg_vol]]
            print(csv_rows)

            with open(os.path.join(args.load_path,"metrics.csv"), "w", newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(csv_header) 
                writer.writerows(csv_rows)
        

