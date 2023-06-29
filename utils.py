
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
import torchio as tio 

CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

def train(model, optimizer, loss_function, eval_function, loader, val_loader, num_epoch, gpu):
    for epoch in range(1, num_epoch+1):
        epoch_losses = train_epoch_complete_case(model, optimizer, loss_function, loader, gpu)
        print(f'EPOCH: {epoch} TRAIN mean loss: {np.mean(epoch_losses).mean():0.3f}')

        epoch_loss_val = validation(model, gpu, eval_function, val_loader)
        print(f'EPOCH: {epoch} Valid mean dice: {np.array(epoch_loss_val).mean():0.3f}')

def train_ssl(model, optimizer, loss_function, eval_function, loader, loader_ulb, val_loader, num_epoch, lmbd, gpu):
    for epoch in range(1, num_epoch+1):
        epoch_losses = train_epoch_SegPL(model, optimizer, loss_function, loader, loader_ulb, lmbd, gpu)
        print(f'EPOCH: {epoch} TRAIN mean loss: {np.mean(epoch_losses).mean():0.3f}')

        epoch_loss_val = validation(model, gpu, eval_function, val_loader)
        print(f'EPOCH: {epoch} Valid mean dice: {np.array(epoch_loss_val).mean():0.3f}')

        
        
def validation(model, gpu, loss_function, loader):
    epoch_losses = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            inputs, targets = prepare_batch(batch, gpu)
            logits = model(inputs)
            batch_loss = loss_function(logits, targets)
            epoch_losses.append(batch_loss.item())
    return(epoch_losses)

def train_epoch_complete_case(model, optimizer, loss_function, loader, gpu):
    epoch_losses = []
    model.train()
    for batch_idx, batch in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        inputs, targets = prepare_batch(batch, gpu)
        #print(inputs.shape, targets.shape)
        logits = model(inputs)
        batch_loss = loss_function(logits, targets)
        batch_loss.backward()
        optimizer.step()
        epoch_losses.append(batch_loss.item())
    return(epoch_losses)

def train_epoch_SegPL(model, optimizer, loss_function, loader, loader_ulb, lmbd, gpu):
    epoch_losses = []
    model.train()
    for batch_idx, (batch, batch_u) in enumerate(tqdm(zip(loader, loader_ulb))):
        optimizer.zero_grad()
        inputs, targets = prepare_batch(batch, gpu)
        inputs_u, _ = prepare_batch(batch_u, gpu)

        logits = model(inputs)
        batch_loss = loss_function(logits, targets)
        
        logits_u = model(inputs_u)
        
        print((logits_u>0).shape)
        
        print((logits_u>0)[:,1:].detach().shape)
        print(torch.unbind((logits_u>0)[:,1].detach())[0].shape)

        pseudo_labels =  torch.stack([tio.OneHot(num_classes=2)(m) for m in torch.unbind((logits_u>0)[:,1].detach(), dim=0) ], dim=0)
        
        unsup_loss = loss_function(logits_u, pseudo_labels)
        
        batch_loss += lmbd * unsup_loss
        
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
            
def get_exams_train_test(data_path, num_eval):
    #patients = os.listdir(data_path)
    patients = list(np.load(os.path.join(data_path+'positive_patients.npy')))
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

