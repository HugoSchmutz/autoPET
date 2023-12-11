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

from monai.data import list_data_collate


import torchio as tio

from utils import net_builder, get_logger, count_parameters, get_ssl_dataset, standard_transform
from SegPL import SegPL
from CompleteCase import CompleteCase
from SegPL_U import SegPL_U
from SegPL_UA import SegPL_UA
from SegPL_MC import SegPL_MC
from train_utils import TBLog

port_dict = {0:{1:8080, 2:8081, 3:8082, 4:8083, 5:8084}, 
            1:{1:8085, 2:8086, 3:8087, 4:8088, 5:8090}, 
            2:{1:8091, 2:8100, 3:8123, 4:8124, 5:8125}, 
            3:{1:8126, 2:8130, 3:8172, 4:8181, 5:8182}, 
            4:{1:8190, 2:8199, 3:8257, 4:8258, 5:8284}}


def find_open_ports():
    for port in range(8080,10002):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        res = sock.connect_ex(('127.0.0.1', port))
        if res != 0:
            yield port
        sock.close()
        
def change_dist_url(args):
    localhost = "tcp://127.0.0.1:"
    #if args.modified_fixmatch:
    #    if args.debiased:
    #        port = str(port_dict[args.seed][3])
    #    else:
    #        if args.ulb_loss_ratio > 0:
    #            port = str(port_dict[args.seed][1])
    #        else:
    #            port = str(port_dict[args.seed][2])
    #else:
    #    if args.ulb_loss_ratio>0:
    #        port =str(port_dict[args.seed][4])
    #    else:
    #        port = str(port_dict[args.seed][5])
    available_ports = list(find_open_ports())
    port = available_ports[0]
    return(localhost+str(port))


def change_run_name(args):
    name = f'_{args.ulb_loss_fct}_{args.num_labels}_{args.ulb_loss_ratio}_{args.seed}_{args.dropout}_{args.learning_rate}'
    if args.count_unselected_pixels:
        name = '_normal' + name
    else:
        name = '_masked' + name
    if args.SegPL_U:
        name = '_softmax' + name
    elif args.UA:
        name = '_UA' + name
    elif args.MC_dropout:
        name = '_MC' + name
    
        
    
    if args.mean_teacher:
        name = 'MT' + name
    elif args.pi_model:
        name = 'PI' + name
    elif args.SegPL or args.SegPL_U or args.MC_dropout:
        name = 'PL' + name
    else:
        name = 'CC' + name

    if args.debiased:
        name = 'De' + name
    if args.finetune:
        name = 'FT_' + name
    return(name)

def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''
    
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume or args.finetune:
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
    args.batch_size = int(args.batch_size / args.world_size)
    
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
    
    global best_acc1
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
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        warnings.warn(f"hello: {save_path}")
        tb_log = TBLog(save_path, 'tensorboard')
        warnings.warn("hello")
        logger_level = "INFO"
    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")

    # SET SegPL: class SegPL 
    _net_builder = net_builder
    
    if args.SegPL:      
        model = SegPL(_net_builder,
                        args.num_classes,
                        args.ema_m,
                        args.p_cutoff,
                        args.ulb_loss_ratio,
                        args.dropout,
                        num_eval_iter=args.num_eval_iter,
                        tb_log=tb_log,
                        logger=logger)
        #Set train losses
        model.set_supervised_loss(args.lb_loss_fct, to_onehot_y=False, softmax=True, include_background=False, batch=True, count_unselected_pixels = False)
        model.set_unsupervised_loss(args.ulb_loss_fct, to_onehot_y=False, softmax=True, include_background=False, batch=True, count_unselected_pixels = False)
    elif args.SegPL_U:      
        model = SegPL_U(_net_builder,
                        args.num_classes,
                        args.ema_m,
                        args.p_cutoff,
                        args.threshold,
                        args.ulb_loss_ratio,
                        args.dropout,
                        num_eval_iter=args.num_eval_iter,
                        tb_log=tb_log,
                        logger=logger)
        #Set train losses
        model.set_supervised_loss(args.lb_loss_fct, to_onehot_y=False, softmax=True, include_background=False, batch=True, count_unselected_pixels = args.count_unselected_pixels)
        model.set_unsupervised_loss(args.ulb_loss_fct, to_onehot_y=False, softmax=True, include_background=False, batch=True, count_unselected_pixels = args.count_unselected_pixels)
    elif args.MC_dropout:
        model = SegPL_MC(_net_builder,
                        args.num_classes,
                        args.ema_m,
                        args.p_cutoff,
                        args.threshold,
                        args.ulb_loss_ratio,
                        args.dropout,
                        num_eval_iter=args.num_eval_iter,
                        tb_log=tb_log,
                        logger=logger)
        #Set train losses
        model.set_supervised_loss(args.lb_loss_fct, to_onehot_y=False, softmax=True, include_background=False, batch=True, count_unselected_pixels = args.count_unselected_pixels)
        model.set_unsupervised_loss(args.ulb_loss_fct, to_onehot_y=False, softmax=True, include_background=False, batch=True, count_unselected_pixels = args.count_unselected_pixels)
    elif args.UA:
        model = SegPL_UA(_net_builder,
                        args.num_classes,
                        args.ema_m,
                        args.p_cutoff,
                        args.threshold,
                        args.ulb_loss_ratio,
                        args.dropout,
                        num_eval_iter=args.num_eval_iter,
                        tb_log=tb_log,
                        logger=logger)
        #Set train losses
        model.set_supervised_loss(args.lb_loss_fct, to_onehot_y=False, softmax=True, include_background=False, batch=True, count_unselected_pixels = args.count_unselected_pixels)
        model.set_unsupervised_loss(args.ulb_loss_fct, to_onehot_y=False, softmax=True, include_background=False, batch=True, count_unselected_pixels = args.count_unselected_pixels)
    else:      
        model = CompleteCase(_net_builder,
                        args.num_classes,
                        args.ema_m,
                        args.dropout,
                        num_eval_iter=args.num_eval_iter,
                        tb_log=tb_log,
                        logger=logger)
        #Set train losses
        model.set_loss(args.lb_loss_fct, to_onehot_y=False, softmax=True, include_background=False, batch=True, count_unselected_pixels=False)

    logger.info(f'Number of Trainable Params: {count_parameters(model.train_model)}')
    
    #Set optimizer
    optimizer = torch.optim.Adam(model.train_model.parameters(), lr=args.learning_rate)
    model.set_optimizer(optimizer)
    
    

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
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model.train_model.cuda(args.gpu)
            model.train_model = torch.nn.parallel.DistributedDataParallel(model.train_model,
                                                                          device_ids=[args.gpu],
                                                                          find_unused_parameters=False)
            model.eval_model.cuda(args.gpu)
            
        else:
            # if arg.gpu is None, DDP will divide and allocate batch_size
            # to all available GPUs if device_ids are not set.
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.train_model = model.train_model.cuda(args.gpu)
        model.eval_model = model.eval_model.cuda(args.gpu)
        
    else:
        model.train_model = torch.nn.DataParallel(model.train_model).cuda()
        model.eval_model = torch.nn.DataParallel(model.eval_model).cuda()
    
    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")
    
    cudnn.benchmark = True

    
    
    # Prepare data loaders
    patch_size = (args.patch_d0, args.patch_d1, args.patch_d2)
    #sampler = tio.data.UniformSampler(patch_size)
    
    # probabilities = {0: 0.5, 1: 0.5}
    # sampler = tio.data.LabelSampler(
    #     patch_size=patch_size,
    #     label_name='segmentation',
    #     label_probabilities=probabilities,
    # )
    
    sampler = tio.data.UniformSampler(
        patch_size=patch_size
    )
    
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
    
    patches_dict['train_lb'] = tio.Queue(
        subjects_dataset=dset_dict['train_lb'],
        max_length=args.max_queue_length,
        samples_per_volume=args.samples_per_volume,
        sampler=sampler,
        num_workers=args.num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    patches_dict['train_ulb'] = tio.Queue(
        subjects_dataset=dset_dict['train_ulb'],
        max_length=args.max_queue_length,
        samples_per_volume=args.samples_per_volume,
        sampler=sampler,
        num_workers=args.num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )



    loader_dict['train_lb'] = torch.utils.data.DataLoader(
        patches_dict['train_lb'], batch_size=args.batch_size)

    loader_dict['train_ulb'] = torch.utils.data.DataLoader(
        patches_dict['train_ulb'], batch_size=args.batch_size * args.uratio)

    loader_dict['eval'] = torch.utils.data.DataLoader(eval_dset, batch_size=1, num_workers=0, collate_fn = list_data_collate)


    ## set DataLoader
    model.set_data_loader(loader_dict)
    
    #If args.resume, load checkpoints from args.load_path
    if args.resume or args.finetune:
        model.load_model(args.load_path)
    if args.finetune: model.it = 0
    # START TRAINING of FixMatch
    trainer = model.train
    trainer(args, logger=logger)
        
    if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        model.save_model('latest_model.pth', save_path)
        
    logging.warning(f"GPU {args.rank} training is FINISHED")
    

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='completecase')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--finetune', action='store_true')

    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')

    '''
    Training Configuration of training
    '''
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--num_train_iter', type=int, default=15000, 
                        help='total number of training iterations')
    parser.add_argument('--num_iteration_finetuning', type=int, default=0, 
                        help='total number of finetuning iterations using DeFixmatch')
    parser.add_argument('--num_eval_iter', type=int, default=500,
                        help='evaluation frequency')
    parser.add_argument('--num_labels', type=int, default=200)
    parser.add_argument('--num_val', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=20,
                        help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=2,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--validation_batch_size', type=int, default=24,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    parser.add_argument('--ulb_loss_ratio', type=float, default=0.05)
    parser.add_argument('--patch_d0', type=int, default=128, help='patch size along the first dimension')
    parser.add_argument('--patch_d1', type=int, default=128, help='patch size along the second dimension')
    parser.add_argument('--patch_d2', type=int, default=32, help='patch size along the third dimension')
    parser.add_argument('--samples_per_volume', type=int, default=10)
    parser.add_argument('--max_queue_length', type=int, default=400)

    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--p_cutoff', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.99, help='ema momentum for eval_model')
    parser.add_argument('--hard_label', type=bool, default=True)

    """
    SSL Configuration
    """

    parser.add_argument('--debiased', action='store_true')
    parser.add_argument('--MC_dropout', action='store_true')
    parser.add_argument('--UA', action='store_true')
    parser.add_argument('--mean_teacher', action='store_true', help='generation of pseudo-labels with mean teacher')
    parser.add_argument('--pi_model', action='store_true', help='pi_model')
    parser.add_argument('--SegPL', action='store_true', help='Segmentation Pseudo Label')
    parser.add_argument('--SegPL_U', action='store_true', help='Segmentation Pseudo Label masked by uncertainty')
    '''
    Optimizer configurations
    '''
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--amp', action='store_true', help='use mixed precision training or not')
    parser.add_argument('--lb_loss_fct', type=str, default='DiceCE', help = 'Choose between DiceCE, Dice, CE')
    parser.add_argument('--ulb_loss_fct', type=str, default='CE', help = 'Choose between DiceCE, Dice, CE, maskedDiceCE, maskedDice, maskedCE')
    parser.add_argument('--count_unselected_pixels', type=bool, default=True, help = 'Do we count masked pixels in the mean of the loss?')
    
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
    parser.add_argument('--data_dir', type=str, default='./data/FDG-PET-CT-Lesions_nifti/')
    parser.add_argument('--patients_list_dir', type=str, default='./data/FDG-PET-CT-Lesions_nifti/')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=20)


    '''
    multi-GPUs & Distrbitued Training
    '''
    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)

    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()
    args.save_name = change_run_name(args)
    args.dist_url = change_dist_url(args)    
    main(args)
