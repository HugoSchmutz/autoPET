from __future__ import print_function, division
import os

import torch
import torch.nn.functional as F
import torchio as tio
from monai.data import list_data_collate

import SimpleITK as sitk
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
from visualise import plot_diff, show_mip_pet_and_mask
from matplotlib.backends.backend_pdf import PdfPages
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose, EnsureType


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
    parser.add_argument('--new_test_set', action = 'store_true', help= 'use the new and bigger test set.')
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
    net.eval()
    
    T_eval = 1
    test_set = get_test_dataset(args.data_dir, args.patients_list_dir, standard_transform, new=args.new_test_set)
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=0, collate_fn = list_data_collate)
 
    dice_loss_fct = monai.losses.DiceLoss(to_onehot_y=False,softmax=True,include_background=False,batch=True)
    
    post_pred = Compose([
        EnsureType(),
        AsDiscrete(argmax=True, to_onehot=2),
    ])


    filename_pdf = os.path.join(args.load_path,  'images_gt_predictions.pdf')
    with torch.no_grad():
        with PdfPages(filename_pdf) as pdf:
            for uid, data in enumerate(tqdm(test_loader)):
                roi_size = (128, 128, 32)
                sw_batch_size = 40
                X, y = prepare_batch(data, args.gpu)

                mean_logits = torch.zeros(X.shape).cuda(args.gpu)
                for i in range(T_eval):
                    mean_logits = +sliding_window_inference(X, roi_size, sw_batch_size, net, mode="gaussian", overlap=0.50)
                mean_logits = mean_logits/T_eval
                #mean_logits = [post_pred(i) for i in decollate_batch(mean_logits)]
                
                mask_out = torch.argmax(mean_logits, dim=1).detach().cpu().numpy().squeeze()
                mask_out = mask_out.astype(np.uint8)               
                
                #csv_rows = [[dice_sc, false_pos_vol, false_neg_vol]]
                #print(csv_rows)

                #with open(os.path.join(args.load_path,"metrics.csv"), "w", newline='') as f:
                #    writer = csv.writer(f, delimiter=',')
                #    writer.writerow(csv_header) 
                #    writer.writerows(csv_rows)
                
                #####################################
                #  plot prediction vs ground-truth  #
                #####################################
                fig, axes = plt.subplots(1, 3, figsize=(15, 10))
                # convert to numpy
                
                pet= tio.ToCanonical()(tio.ScalarImage(os.path.join(data['path'][0], 'SUV.nii')))
                pet_array = np.array(pet[tio.DATA][0]).T
                pred_mask_array = mask_out.T
                gt_mask_array = np.array(data['segmentation'][tio.DATA][0,1]).T
                
                suv_max = (np.flip(pet_array, axis=0)*mask_out.T).max()
                true_suv_max = (np.flip(pet_array, axis=0)*np.array(data['segmentation'][tio.DATA][0,1].T)).max()
                #print((np.flip(pet_array, axis=0)*mask_out.T).argmax(), (np.flip(pet_array, axis=0)*np.array(data['segmentation'][tio.DATA][0,1].T)).argmax())
                #print((np.flip(pet_array, axis=0)*mask_out.T).max(), (np.flip(pet_array, axis=0)*np.array(data['segmentation'][tio.DATA][0,1].T)).max())
                mse_suv_max = (suv_max - true_suv_max.item())**2/(true_suv_max.item()+10**(-6))**2
                
                pet_spacing = np.array([3.,2.03642, 2.03642     ])
                # MIP with ground truth
                show_mip_pet_and_mask(pet_array=pet_array,
                                    mask_array=gt_mask_array,
                                    axis=1,
                                    ax=axes[0]
                                    )
                aspect = [pet_spacing[ii] for ii in range(3) if ii != 1]
                axes[0].set_aspect(aspect[0]/aspect[1])
                # MIP coronal with prediction
                plot_diff(pet_array, np.zeros(gt_mask_array.shape), pred_mask_array,
                        axis=1, ax=axes[1], title='')
                # aspect = [pet_spacing[ii] for ii in range(3) if ii != 1]
                axes[1].set_aspect(aspect[0]/aspect[1])
                # MIP coronal empty
                plot_diff(pet_array, np.zeros(gt_mask_array.shape), np.zeros(pred_mask_array.shape),
                        axis=1, ax=axes[2], title='')
                aspect = [pet_spacing[ii] for ii in range(3) if ii != 2]
                axes[2].set_aspect(aspect[0]/aspect[1])
                suptitle = ('Ground Truth and Detections\n'
                            'GT=green, pred=red\n'
                            'id: {0}'.format(uid))
                fig.suptitle(suptitle)
                fig.tight_layout()
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()