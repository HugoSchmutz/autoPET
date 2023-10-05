from __future__ import print_function, division
import os

import torch
import torch.nn.functional as F
import torchio as tio
from monai.data import list_data_collate

from monai.inferers import sliding_window_inference
from utils import net_builder, get_test_dataset, standard_transform, prepare_batch, compute_metrics, voxel_vol
import numpy as np
import nibabel as nib
import pathlib as plb
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import monai

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
    parser.add_argument('--patients_list_dir', type=str, default='./data/FDG-PET-CT-Lesions_nifti/')
    parser.add_argument('--num_classes', type=int, default=2)
    
    '''
    GPU
    '''
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    
    args = parser.parse_args()
    
    print(args.load_path)
    checkpoint_path = os.path.join(args.load_path, 'model_best.pth')
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['train_model'] if args.use_train_model else checkpoint['eval_model']
    
    _net_builder = net_builder
    net = _net_builder(num_classes=args.num_classes)
    net.load_state_dict(load_model)
    
    if torch.cuda.is_available():
        net.cuda()
    net.train()
    
    T_eval = 10
    test_set = get_test_dataset(args.data_dir, args.patients_list_dir, standard_transform)
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=0, collate_fn = list_data_collate)
 
    dice_loss_fct = monai.losses.DiceLoss(to_onehot_y=False,softmax=True,include_background=False,batch=True)

    metrics = []    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            roi_size = (128, 128, 32)
            sw_batch_size = 40
            
            X, y = prepare_batch(data, args.gpu)
            
            mean_logits = torch.zeros(X.shape).cuda(args.gpu)
            for i in range(T_eval):
                mean_logits = +sliding_window_inference(X, roi_size, sw_batch_size, net, mode="gaussian", overlap=0.50)
            mean_logits = mean_logits/T_eval
            print(mean_logits.shape)
            mask_out = torch.argmax(mean_logits, dim=1).detach().cpu().numpy().squeeze()
            mask_out = mask_out.astype(np.uint8)               
            
            
            dice_loss = dice_loss_fct(mean_logits, y).detach().cpu().item()
            
            
            predicted_tumour_volume = mask_out.sum() * voxel_vol
            true_volume = data['segmentation'][tio.DATA][0,1].sum() * voxel_vol
            mse_volume = (predicted_tumour_volume - true_volume)**2
            
            dice_sc, false_pos_vol, false_neg_vol, true_nb_lesions, pred_nb_lesions = compute_metrics(mask_out, data['segmentation'][tio.DATA][0,1])
            metrics.append([dice_sc, false_pos_vol, false_neg_vol, predicted_tumour_volume, true_volume, mse_volume, true_nb_lesions, pred_nb_lesions, dice_loss])
            
            #csv_rows = [[dice_sc, false_pos_vol, false_neg_vol]]
            #print(csv_rows)

            #with open(os.path.join(args.load_path,"metrics.csv"), "w", newline='') as f:
            #    writer = csv.writer(f, delimiter=',')
            #    writer.writerow(csv_header) 
            #    writer.writerows(csv_rows)
    mean_dice_sc = np.mean(metrics, axis=0)[0]
    total_false_pos_vol = np.sum(metrics, axis=0)[1]
    total_false_negvol = np.mean(metrics, axis=0)[2]
    mean_mse_volume = np.mean(metrics, axis=0)[5]
    mean_dice_loss = np.mean(metrics, axis=0)[8]
    
    print(f'Mean Dice: {mean_dice_sc:0.3f}, Mean Dice Loss: {dice_loss:0.3f}, False positive: {total_false_pos_vol:0.3f}, False negative: {total_false_negvol:0.3f}, Mean MSE volume: {mean_mse_volume:0.3f}')
    
    metrics = np.array(metrics)
    res= pd.DataFrame({'dice_sc':metrics[:,0],
                      'false_pos_vol':metrics[:,1], 
                      'false_neg_vol':metrics[:,2],
                      'predicted_volume':metrics[:,3],
                      'true_volume':metrics[:,4],
                      'MSE':metrics[:,5],
                      'true_nb_lesions': metrics[:,6], 
                      'pred_nb_lesions': metrics[:,7]
                        })
    
    res.to_csv(os.path.join(args.load_path,"metrics.csv"))
    
    plt.figure()
    sns.histplot(data=res, x = 'false_pos_vol')
    plt.savefig(os.path.join(args.load_path,'false_pos_vol.pdf'), format = 'pdf')
    
    plt.figure()
    sns.histplot(data=res, x = 'false_neg_vol')
    plt.savefig(os.path.join(args.load_path,'false_neg_vol.pdf'), format = 'pdf')
    
    plt.figure()
    sns.histplot(data=res, x = 'dice_sc')
    plt.savefig(os.path.join(args.load_path,'dice_sc.pdf'), format = 'pdf')
    
    plt.figure()
    sns.regplot(x="true_volume", y="predicted_volume", data=res)
    plt.title(np.corrcoef(res['predicted_volume'].values, res['true_volume'].values)[0,1])
    plt.savefig(os.path.join(args.load_path,'volume.pdf'), format = 'pdf')
    print(np.corrcoef(res['predicted_volume'].values, res['true_volume'].values))
    
    plt.figure()
    sns.regplot(x="true_nb_lesions", y="pred_nb_lesions", data=res)
    plt.title(np.corrcoef(res['true_nb_lesions'].values, res['pred_nb_lesions'].values)[0,1])
    plt.savefig(os.path.join(args.load_path,'nb_lesions.pdf'), format = 'pdf')
    print(np.corrcoef(res['true_nb_lesions'].values, res['pred_nb_lesions'].values))
    
    