
import torch
import torchvision
import torchio as tio
import torch.nn.functional as F
import enum
import time
import numpy as np
from unet import UNet
from tqdm.auto import tqdm
import os
from sklearn.model_selection import train_test_split
import monai
import logging
import nibabel as nib
import pathlib as plb
import cc3d
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist
from typing import Any
import pickle

voxel_vol = 0.012441020965576172

CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4    

standard_transform = tio.Compose([
        tio.RescaleIntensity(include=['ct'], in_min_max = (-100, 250), out_min_max = (0,1)),
        tio.RescaleIntensity(include=['pet'], in_min_max = (0, 15), out_min_max = (0,1)),
        tio.OneHot(num_classes=2)])

def train(model, optimizer, loss_function, eval_function, loader, val_loader, num_epoch, gpu):
    for epoch in range(1, num_epoch+1):
        epoch_losses = train_epoch_complete_case(model, optimizer, loss_function, loader, gpu)
        print(f'EPOCH: {epoch} TRAIN mean loss: {np.mean(epoch_losses).mean():0.3f}')

        if epoch % 10 == 0:
            epoch_loss_val, metrics = validation(model, gpu, eval_function, val_loader)
            #dice_sc, false_pos_vol, false_neg_vol = np.mean(metrics, axis = 0)
            #print(f'Valid mean dice: {np.array(epoch_loss_val).mean():0.3f}, DC: {dice_sc:0.3f}, FP: {false_pos_vol:0.3f}, FN: {false_neg_vol:0.3f}')
            print(f'Valid mean dice: {np.array(epoch_loss_val).mean():0.3f}')
        
def train_ssl(model, optimizer, loss_function, ulb_loss_function, eval_function, loader, loader_ulb, val_loader, num_epoch, lmbd, gpu, num_epoch_pretraining):
    for epoch in range(1, num_epoch+1):
        if epoch < num_epoch_pretraining:
            epoch_losses = train_epoch_complete_case(model, optimizer, loss_function, loader, gpu)
        else:
            epoch_losses = train_epoch_SegPL(model, optimizer, loss_function, ulb_loss_function, loader, loader_ulb, lmbd, gpu)
        print(f'EPOCH: {epoch} TRAIN mean loss: {np.mean(epoch_losses).mean():0.3f}')
        if epoch % 10 == 0:
            epoch_loss_val, metrics = validation(model, gpu, eval_function, val_loader)
            #dice_sc, false_pos_vol, false_neg_vol = np.mean(metrics, axis = 0)
            #print(f'Valid mean dice: {np.array(epoch_loss_val).mean():0.3f}, DC: {dice_sc:0.3f}, FP: {false_pos_vol:0.3f}, FN: {false_neg_vol:0.3f}')
            print(f'Valid mean dice: {np.array(epoch_loss_val).mean():0.3f}')



class MaskedDiceCELoss(monai.losses.DiceCELoss):
    """
    Add an additional `masking` process before `DiceLoss`, accept a binary mask ([0, 1]) indicating a region,
    `input` and `target` will be masked by the region: region with mask `1` will keep the original value,
    region with `0` mask will be converted to `0`. Then feed `input` and `target` to normal `DiceLoss` computation.
    This has the effect of ensuring only the masked region contributes to the loss computation and
    hence gradient calculation.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Args follow :py:class:`monai.losses.DiceLoss`.
        """
        super().__init__(*args, **kwargs)
        self.spatial_weighted = monai.losses.MaskedLoss(loss=super().forward)


    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor ) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should B1H[WD] or 11H[WD].
        """
        return self.spatial_weighted(input=input, target=target, mask=mask)  # type: ignore[no-any-return]

   
def set_loss(loss_name, to_onehot_y, softmax, include_background, batch):
    if loss_name == 'Dice':
        loss = monai.losses.DiceLoss(to_onehot_y=to_onehot_y, softmax=softmax, include_background=include_background, batch=batch)
    elif loss_name == 'DiceCE':
        loss = monai.losses.DiceCELoss(to_onehot_y=to_onehot_y, softmax=softmax, include_background=include_background, batch=batch)
    elif loss_name == 'CE':
        loss = monai.losses.DiceCELoss(to_onehot_y=to_onehot_y, softmax=softmax, include_background=include_background, batch=batch, lambda_dice=1.0)
    elif loss_name == 'maskedCE':
        loss = MaskedDiceCELoss(to_onehot_y=to_onehot_y, softmax=softmax, include_background=include_background, batch=batch, lambda_dice=1.0)
    elif loss_name == 'maskedDiceCE':
        loss = MaskedDiceCELoss(to_onehot_y=to_onehot_y, softmax=softmax, include_background=include_background, batch=batch)
    elif loss_name == 'maskedDice':
        loss = monai.losses.MaskedDiceLoss(to_onehot_y=to_onehot_y, softmax=softmax, include_background=include_background, batch=batch)
    return(loss)   
              
def validation(model, gpu, loss_function, loader):
    epoch_losses = []
    metrics = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            inputs, targets = prepare_batch(batch, gpu)
            logits = model(inputs)
            batch_loss = loss_function(logits, targets)
            #metrics += [compute_metrics(l, p) for l,p in zip(targets[:,1].cpu(), (logits>0).long()[:,1].cpu())]
            epoch_losses.append(batch_loss.item())
    return(epoch_losses, metrics)

def train_epoch_complete_case(model, optimizer, loss_function, loader, gpu):
    epoch_losses = []
    model.train()
    for batch_idx, batch in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        inputs, targets = prepare_batch(batch, gpu)
        #print(inputs.shape, targets.shape)
        logits = model(inputs)
        if targets.sum()>0:
            batch_loss = loss_function(logits, targets)
        else:
            torch.zeros(1).cuda()
        batch_loss.backward()
        optimizer.step()
        epoch_losses.append(batch_loss.item())
    return(epoch_losses)

def train_epoch_SegPL(model, optimizer, loss_function, ulb_loss_function, loader, loader_ulb, lmbd, gpu):
    epoch_losses = []
    model.train()
    for batch_idx, (batch, batch_u) in enumerate(tqdm(zip(loader, loader_ulb))):
        optimizer.zero_grad()
        inputs, targets = prepare_batch(batch, gpu)
        inputs_u, _ = prepare_batch(batch_u, gpu)

        #Labelled loss
        logits = model(inputs)
        if targets.sum()>0:
            sup_loss = loss_function(logits, targets)
        else:
            torch.zeros(1).cuda()
        
        #Unlabelled loss
        logits_u = model(inputs_u)
        probabilities = torch.nn.Softmax(dim=1)(logits_u)
        pseudo_labels =  (probabilities>0.5).float().detach()
        mask = (probabilities[:,1]>0.95).float() + (probabilities[:,0]>0.95)
        if pseudo_labels.sum()>0:
            unsup_loss = ulb_loss_function(logits_u, pseudo_labels, mask)
        else:
            torch.zeros(1).cuda()
        batch_loss = sup_loss + lmbd * unsup_loss
        
        batch_loss.backward()
        optimizer.step()
        epoch_losses.append(batch_loss.item())
    return(epoch_losses)

def get_exams(data_path, patients_list):
    paths_list = []
    for patient in patients_list:
        examens = os.listdir(os.path.join(data_path,patient))
        for exam in examens:
            paths_list.append(os.path.join(os.path.join(data_path, patient), exam))
    return(paths_list)
            
def get_exams_train(data_path, patients_list_dir, num_eval):
    #patients = os.listdir(data_path)
    patients = list(np.load(os.path.join(patients_list_dir,'positive_patients_train.npy')))
    patients_train, patients_test = train_test_split(patients, test_size = num_eval, random_state=0)
    all_paths_train, all_paths_test = get_exams(data_path, patients_train), get_exams(data_path,patients_test)

    return(all_paths_train, all_paths_test)
    
def subjects_list(exams):
    subjects = []
    for path in exams:
        subject = tio.Subject(
            ct=tio.ScalarImage(os.path.join(path, 'CTres.nii.gz') ),
            pet=tio.ScalarImage(os.path.join(path, 'SUV.nii.gz') ),
            segmentation=tio.LabelMap(os.path.join(path, 'SEG.nii.gz') ),
        )
        subjects.append(subject)
    return(subjects)


def split_labelled_unlabelled(subjects, nl):
    n_train = len(subjects)
    nu = n_train - nl
    num_split_subjects = nl, nu
    labelled_subjects, unlabelled_subjects = torch.utils.data.random_split(subjects, num_split_subjects)
    return(labelled_subjects, unlabelled_subjects)

class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'

def prepare_batch(batch, gpu):
    pet = batch['pet'][tio.DATA].cuda(gpu)
    ct = batch['ct'][tio.DATA].cuda(gpu)
    inputs = torch.cat([pet,ct], axis=CHANNELS_DIMENSION)
    targets = batch['segmentation'][tio.DATA].cuda(gpu)
    return inputs, targets

def get_dice_score(output, target, epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score

def get_dice_loss(output, target):
    return 1 - get_dice_score(output, target)

class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value

def net_builder(num_classes=2, dropout = 0):
    unet = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm= monai.networks.layers.Norm.BATCH,
        dropout = dropout,
    )
    return unet

    
def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    
    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    
    return logger
            
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_exams_test(data_path, patients_list_dir):
    patients = list(np.load(os.path.join(patients_list_dir,'positive_patients_test.npy')))
    all_paths_test = get_exams(data_path, patients)
    return(all_paths_test)

def get_test_dataset(data_path, patients_list_dir, transform):
    all_paths_test = get_exams_test(data_path, patients_list_dir)
    subjects_test = subjects_list(all_paths_test)
    
    test_set = tio.SubjectsDataset(
        subjects_test, transform=transform)
    return(test_set)


def get_ssl_dataset(data_path, patients_list_dir, num_labelled, num_eval, transform=standard_transform):
    all_paths_train, all_paths_eval = get_exams_train(data_path, patients_list_dir, num_eval)
    subjects_train = subjects_list(all_paths_train)
    subjects_eval = subjects_list(all_paths_eval)



    labelled_subjects, unlabelled_subjects = split_labelled_unlabelled(subjects_train, num_labelled)

    training_labelled_set = tio.SubjectsDataset(
        labelled_subjects, transform=transform)

    training_unlabelled_set = tio.SubjectsDataset(
        unlabelled_subjects, transform=transform)

    validation_set = tio.SubjectsDataset(
        subjects_eval, transform=transform)
    return(training_labelled_set, training_unlabelled_set, validation_set)



def setattr_cls_from_kwargs(cls, kwargs):
    #if default values are in the cls,
    #overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls,key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])

        
def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = 'hello'
    test_cls = _test_cls()
    config = {'a': 3, 'b': 'change_hello', 'c':5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")
        
 
 
def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    mask_nii = nib.load(str(nii_path))
    mask = mask_nii.get_fdata()
    pixdim = mask_nii.header['pixdim']   
    voxel_vol = pixdim[1]*pixdim[2]*pixdim[3]/1000
    return mask, voxel_vol



def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def false_pos_pix(gt_array,pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    pred_nb_lesions = pred_conn_comp.max()
    false_pos = 0
    for idx in range(1,pred_conn_comp.max()+1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask*gt_array).sum() == 0:
            false_pos = false_pos+comp_mask.sum()
    return false_pos, pred_nb_lesions



def false_neg_pix(gt_array,pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    true_nb_lesions = gt_conn_comp.max()
    false_neg = 0
    for idx in range(1,gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask*pred_array).sum() == 0:
            false_neg = false_neg+comp_mask.sum()
            
    return false_neg, true_nb_lesions


def dice_score(mask1,mask2):
    # compute foreground Dice coefficient
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum
    return dice_score



def compute_metrics(prediction, label):
    # main function
    gt_array = label.numpy()
    pred_array = prediction

    false_neg_vol, true_nb_lesions = false_neg_pix(gt_array, pred_array)
    false_pos_vol, pred_nb_lesions = false_pos_pix(gt_array, pred_array)
    dice_sc = dice_score(gt_array,pred_array)
    return dice_sc, false_pos_vol*voxel_vol, false_neg_vol*voxel_vol, true_nb_lesions, pred_nb_lesions

