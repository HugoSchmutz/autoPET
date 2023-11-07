from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction

import warnings
from collections.abc import Callable, Sequence
from typing import Any
import warnings
import monai
from monai.networks import one_hot
from typing import Any



class MSELoss(_Loss):
    def __init__(self, include_background: bool= True, to_onehot_y: bool = False, softmax: bool = True,
                 reduction = LossReduction.MEAN, sigmoid: bool = False, batch: bool = False,*args: Any, **kwargs: Any) -> None:
        """
        Args follow :py:class:`monai.losses.DiceLoss`.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.batch = batch
        self.loss = nn.MSELoss(reduction='none')
        
        
        
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should B1H[WD] or 11H[WD].
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
                
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
                
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")


        f = self.loss(input, target)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            f=f
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f
    
    
class MaskedMSELoss(_Loss):
    def __init__(self, include_background: bool= True, to_onehot_y: bool = False, softmax: bool = True, count_unselected_pixels: bool = True,
                 reduction = LossReduction.MEAN, sigmoid: bool = False, batch: bool = False,*args: Any, **kwargs: Any) -> None:
        """
        Args follow :py:class:`monai.losses.DiceLoss`.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.batch = batch
        self.count_unselected_pixels = count_unselected_pixels
        self.loss = nn.MSELoss(reduction='none')
        
        
        
        
    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should B1H[WD] or 11H[WD].
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
                
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
                
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        if mask.shape[1]==1:
            mask = mask[:,0]
        
        f = self.loss(input, target)
        f=f[:,0]
        if self.count_unselected_pixels:
            f = torch.mean(f*mask,  dim = (1,2,3))
        else:
            f = torch.sum(f*mask,  dim = (1,2,3))/torch.sum(mask, dim = (1,2,3))

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            f=f
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f



class CELoss(_Loss):
    def __init__(self, to_onehot_y: bool = False, 
                 reduction = LossReduction.MEAN, 
                 batch: bool = False,*args: Any, **kwargs: Any) -> None:
        """
        Args follow :py:class:`monai.losses.DiceLoss`.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        
        self.to_onehot_y = to_onehot_y
        self.batch = batch
        self.loss = nn.CrossEntropyLoss(reduction='none')
        
        
        
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should B1H[WD] or 11H[WD].
        """
        n_pred_ch = input.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
                
                
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        
        f = self.loss(input, target)
        
        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            f=f
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f
    
    
class MaskedCELoss(_Loss):
    def __init__(self, to_onehot_y: bool = False, count_unselected_pixels: bool = True, reduction = LossReduction.MEAN, 
                 batch: bool = False,*args: Any, **kwargs: Any) -> None:
        """
        Args follow :py:class:`monai.losses.DiceLoss`.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        
        self.to_onehot_y = to_onehot_y
        self.batch = batch
        self.count_unselected_pixels = count_unselected_pixels
        self.loss = nn.CrossEntropyLoss(reduction='none')
        
        
        
        
    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should B1H[WD] or 11H[WD].
        """
        n_pred_ch = input.shape[1]
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
                
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        if mask.shape[1]==1:
            mask = mask[:,0]
        
        
        f = self.loss(input, target)
        if self.count_unselected_pixels:
            f = torch.mean(f*mask,  dim = (1,2,3))
        else:
            f = torch.sum(f*mask,  dim = (1,2,3))/torch.sum(mask, dim = (1,2,3))

        
        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            f=f
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


# class MaskedDiceCELoss(_Loss):
#     def __init__(self, include_background: bool= True, to_onehot_y: bool = False, softmax: bool = True, count_unselected_pixels: bool = True,
#                  reduction = LossReduction.MEAN, sigmoid: bool = False, batch: bool = False,*args: Any, **kwargs: Any) -> None:
#         """
#         Args follow :py:class:`monai.losses.DiceLoss`.
#         """
#         super().__init__(reduction=LossReduction(reduction).value)
        
#         self.to_onehot_y = to_onehot_y
#         self.batch = batch
#         self.count_unselected_pixels = count_unselected_pixels
#         self.loss_ce = nn.CrossEntropyLoss(reduction='none')
#         self.loss_dice = monai.losses.DiceLoss(to_onehot_y=False,softmax=True,include_background=False,batch=False, reduction='none')
        

        
        
        
#     def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             input: the shape should be BNH[WD].
#             target: the shape should be BNH[WD].
#             mask: the shape should B1H[WD] or 11H[WD].
#         """
#         n_pred_ch = input.shape[1]
#         if self.to_onehot_y:
#             if n_pred_ch == 1:
#                 warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
#             else:
#                 target = one_hot(target, num_classes=n_pred_ch)
                
#         if target.shape != input.shape:
#             raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        
#         f_dice = self.loss_dice(input*mask, target*mask).flatten()

#         if mask.shape[1]==1:
#             mask = mask[:,0]
        
#         f_ce = self.loss_ce(input, target)
#         if self.count_unselected_pixels:
#             f_ce = torch.mean(f_ce*mask,  dim = (1,2,3))
#         else:
#             f_ce = torch.sum(f_ce*mask,  dim = (1,2,3))/torch.sum(mask, dim = (1,2,3))

#         f = f_dice + f_ce
        
#         if self.reduction == LossReduction.MEAN.value:
#             f = torch.mean(f)  # the batch and channel average
#         elif self.reduction == LossReduction.SUM.value:
#             f = torch.sum(f) + torch.sum(f_dice)  # sum over the batch and channel dims
#         elif self.reduction == LossReduction.NONE.value:
#             # If we are not computing voxelwise loss components at least
#             # make sure a none reduction maintains a broadcastable shape
#             f=f
#         else:
#             raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

#         return (1/2) * f


from monai.losses.spatial_mask import MaskedLoss
from monai.losses import DiceCELoss
class MaskedDiceCELoss(DiceCELoss):
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
        self.spatial_weighted = MaskedLoss(loss=super().forward)


    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should B1H[WD] or 11H[WD].
        """
        return self.spatial_weighted(input=input, target=target, mask=mask)  # type: ignore[no-any-return]

