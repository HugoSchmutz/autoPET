import enum
import time
import random
import multiprocessing
from pathlib import Path
import os 
import monai
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist

import torchvision
import torchio as tio
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from utils import *
import numpy as np
from unet import UNet
from scipy import stats
import matplotlib.pyplot as plt

from IPython import display
from tqdm.auto import tqdm



def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''
    
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')
        
    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    #distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed 
    ngpus_per_node = torch.cuda.device_count() # number of gpus of each node
    
    #divide the batch_size according to the number of nodes
    args.world_size=1

    args.training_batch_size = int(args.training_batch_size / args.world_size)
    
    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size 
        
        #args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)) 
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''
    
    global best_acc
    args.gpu = gpu
    
    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = False
    
    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu # compute global rank
        
        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    #SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
                           
    
    # Prepare data loaders
    standard_transform = tio.Compose([
        tio.ToCanonical(),
        tio.RescaleIntensity(include=['ct'], in_min_max = (-100, 250), out_min_max = (0,1)),
        tio.RescaleIntensity(include=['pet'], in_min_max = (0, 15), out_min_max = (0,1)),
        #tio.Resample(2),
        tio.OneHot(num_classes=2)])

    patch_size = (args.patch_d0, args.patch_d1, args.patch_d2)
    sampler = tio.data.UniformSampler(patch_size)

    all_paths_train, all_paths_test = get_exams_train_test(args.data_path, args.num_eval)
    subjects_train = subjects_list(all_paths_train)
    subjects_test = subjects_list(all_paths_test)



    labelled_subjects, unlabelled_subjects = split_labelled_unlabelled(subjects_train, args.num_labelled)

    training_labelled_set = tio.SubjectsDataset(
        labelled_subjects, transform=standard_transform)

    training_unlabelled_set = tio.SubjectsDataset(
        unlabelled_subjects, transform=standard_transform)

    validation_set = tio.SubjectsDataset(
        subjects_test, transform=standard_transform)

    print('Training labelled set:', len(training_labelled_set), 'subjects \t Training unlabelled set:', len(training_unlabelled_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')

    patches_training_labelled_set = tio.Queue(
        subjects_dataset=training_labelled_set,
        max_length=args.max_queue_length,
        samples_per_volume=args.samples_per_volume,
        sampler=sampler,
        num_workers=args.num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    patches_training_unlabelled_set = tio.Queue(
        subjects_dataset=training_unlabelled_set,
        max_length=args.max_queue_length,
        samples_per_volume=args.samples_per_volume,
        sampler=sampler,
        num_workers=args.num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    patches_validation_set = tio.Queue(
        subjects_dataset=validation_set,
        max_length=args.max_queue_length,
        samples_per_volume=args.samples_per_volume,
        sampler=sampler,
        num_workers=args.num_workers,
        shuffle_subjects=False,
        shuffle_patches=False,
    )

    training_labelled_loader_patches = torch.utils.data.DataLoader(
        patches_training_labelled_set, batch_size=args.training_batch_size)

    training_unlabelled_loader_patches = torch.utils.data.DataLoader(
        patches_training_unlabelled_set, batch_size=args.training_batch_size)

    validation_loader_patches = torch.utils.data.DataLoader(
        patches_validation_set, batch_size=args.validation_batch_size)



    #Set model, loss and optimizer
    unet = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm= monai.networks.layers.Norm.BATCH,
    )

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate)
    loss_function = monai.losses.DiceCELoss(to_onehot_y=False,softmax=True,include_background=False,batch=True)
    eval_function = monai.losses.DiceCELoss(to_onehot_y=False,softmax=True,include_background=False,batch=True)
    ulb_loss_function = monai.losses.DiceCELoss(to_onehot_y=True,softmax=True,include_background=False,batch=True)

    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            
            '''
            batch_size: batch_size per node -> batch_size per gpu
            workers: workers per node -> workers per gpu
            '''
            args.training_batch_size = int(args.training_batch_size / ngpus_per_node)
            unet.cuda(args.gpu)
            unet = torch.nn.parallel.DistributedDataParallel(unet,
                                                            device_ids=[args.gpu],
                                                            find_unused_parameters=True)            
        else:
            # if arg.gpu is None, DDP will divide and allocate batch_size
            # to all available GPUs if device_ids are not set.
            unet.cuda()
            unet = torch.nn.parallel.DistributedDataParallel(unet)
            
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        unet = unet.cuda(args.gpu)
        
    else:
        unet = torch.nn.DataParallel(unet).cuda()
    
    
    cudnn.benchmark = True

    if args.SegPL:
        train_ssl(unet, optimizer, loss_function, ulb_loss_function, eval_function, training_labelled_loader_patches, 
                training_unlabelled_loader_patches, validation_loader_patches, args.epoch, args.lmbd, args.gpu)
    else:
        train(unet, optimizer, loss_function, eval_function, training_labelled_loader_patches, 
            validation_loader_patches, args.epoch, args.gpu )
    
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='')

    CHANNELS_DIMENSION = 1
    SPATIAL_DIMENSIONS = 2, 3, 4



    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='completecase')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')

    '''
    Training Configuration of training
    '''
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=800)
    parser.add_argument('--num_labelled', type=int, default=200)
    parser.add_argument('--num_eval', type=int, default=50)
    parser.add_argument('--num_test', type = int, default= 50)
    parser.add_argument('--training_batch_size', type=int, default=12,
                        help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=7,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--validation_batch_size', type=int, default=24,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    parser.add_argument('--patch_d0', type=int, default=128, help='patch size along the first dimension')
    parser.add_argument('--patch_d1', type=int, default=128, help='patch size along the second dimension')
    parser.add_argument('--patch_d2', type=int, default=32, help='patch size along the third dimension')
    parser.add_argument('--samples_per_volume', type=int, default=10)
    parser.add_argument('--max_queue_length', type=int, default=500)
    parser.add_argument('--SegPL', action='store_true', help='Segmentation Pseudo Label')
    parser.add_argument('--lmbd', type=float, default = 1., help= 'Unlabelled loss weight')

    '''
    Optimizer configurations
    '''
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)


    '''
    Data Configurations
    '''
    parser.add_argument('--data_path', type=str, default='./data/FDG-PET-CT-Lesions_nifti/')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)


    '''
    multi-GPUs & Distrbitued Training
    '''
    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()
    main(args)
