#import needed library
import os
import logging
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import socket
from tqdm import tqdm
from monai.data import list_data_collate
import pandas as pd

import torchio as tio

from utils import net_builder, get_logger, prepare_batch, get_ssl_dataset, standard_transform, set_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='./saved_models/CC_200_0.0.0/')
    parser.add_argument('--use_train_model', action='store_true')
    
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
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

    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='./data/FDG-PET-CT-Lesions_nifti/')
    parser.add_argument('--patients_list_dir', type=str, default='./data/FDG-PET-CT-Lesions_nifti/')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_labels', type=int, default=200)
    parser.add_argument('--num_val', type=int, default=20)
    
    parser.add_argument('--patch_d0', type=int, default=128, help='patch size along the first dimension')
    parser.add_argument('--patch_d1', type=int, default=128, help='patch size along the second dimension')
    parser.add_argument('--patch_d2', type=int, default=32, help='patch size along the third dimension')
    parser.add_argument('--samples_per_volume', type=int, default=10)
    parser.add_argument('--max_queue_length', type=int, default=400)
    '''
    GPU
    '''
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--lb_loss_fct', type=str, default='DiceCE', help = 'Choose between DiceCE, Dice, CE')
    parser.add_argument('--ulb_loss_fct', type=str, default='DiceCE', help = 'Choose between DiceCE, Dice, CE, maskedDiceCE, maskedDice, maskedCE')
    parser.add_argument('--count_unselected_pixels', type=bool, default=True, help = 'Do we count masked pixels in the mean of the loss?')
    args = parser.parse_args()
    

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = False
    
    print(args.load_path)
    checkpoint_path = os.path.join(args.load_path, 'model_best.pth')
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['train_model']
    
    _net_builder = net_builder
    net = _net_builder(num_classes=args.num_classes, dropout=args.dropout)
    net.load_state_dict(load_model)
    
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    # Prepare data loaders
    patch_size = (args.patch_d0, args.patch_d1, args.patch_d2)
    sampler = tio.data.UniformSampler(patch_size)

    lb_dset, ulb_dset, eval_dset = get_ssl_dataset(args.data_dir, 
                                                   args.patients_list_dir,
                                                   args.num_labels,
                                                   args.num_val, 
                                                   transform=standard_transform)

    
    print('Training labelled set:', len(lb_dset), 'subjects \t Training unlabelled set:', len(ulb_dset), 'subjects')
    print('Validation set:', len(eval_dset), 'subjects')
    
    loader_dict = {}
    patches_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}
    

    patches_dict['train_ulb'] = tio.Queue(
        subjects_dataset=dset_dict['train_ulb'],
        max_length=args.max_queue_length,
        samples_per_volume=args.samples_per_volume,
        sampler=sampler,
        num_workers=args.num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    loader_dict['train_ulb'] = torch.utils.data.DataLoader(
        patches_dict['train_ulb'], batch_size=1)
    
    
    supervised_loss = set_loss(args.lb_loss_fct, to_onehot_y=False, softmax=True, include_background=False, batch=False, count_unselected_pixels = args.count_unselected_pixels)
    unsupervised_loss = set_loss(args.ulb_loss_fct, to_onehot_y=False, softmax=True, include_background=False, batch=False, count_unselected_pixels = args.count_unselected_pixels)
    
    
    losses = {'supervised':[], 'pseudo-label':[]}
    for i, batch in enumerate(tqdm(loader_dict['train_ulb'])):
                
        inputs, y = prepare_batch(batch, args.gpu)
        logits = net(inputs) 
        # Supervised loss
        sup_loss = supervised_loss(logits, y)
        losses['supervised'].append(sup_loss.item())
        
        #Pseudo-label
        probabilities = torch.nn.Softmax(dim=1)(logits)        
        pseudo_labels = (probabilities>0.5).float().detach()
        unsup_loss = unsupervised_loss(logits, pseudo_labels)
        losses['pseudo-label'].append(unsup_loss.item())
        
        
    losses = pd.DataFrame(losses)
    losses.to_csv(f'losses_{args.num_labels}.csv')