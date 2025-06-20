# -*- coding: utf-8 -*-
# Loss functions (PyTorch and own defined)
#
# Own defined loss functions:
# xentropy_loss, dice_loss, mse_loss and msge_loss (https://github.com/vqdang/hover_net)
# WeightedBaseLoss, MAEWeighted, MSEWeighted, BCEWeighted, CEWeighted (https://github.com/okunator/cellseg_models.pytorch)
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

# Bending Loss included
# SAMS-Net Loss included


import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch import nn
from torch.nn.modules.loss import _Loss
from base_ml.base_utils import filter2D, gaussian_kernel2d
import numpy as np

# Imports for SAMS-Net Loss------------
from scipy.ndimage import distance_transform_edt as distance_transform_edt_spicy
from scipy.ndimage import distance_transform_cdt as distance_transform_cdt_spicy
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from distmap import euclidean_distance_transform
#from cucim.core.operations.morphology import distance_transform_edt as distance_transform_edt_cucim
from torchvision.utils import save_image
from skimage.color import rgb2hed, hed2rgb
#--------------------------------------

#from cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor
#import cv2
#import scipy.ndimage as ndimage
#from skimage.morphology import remove_small_objects
#from skimage.segmentation import watershed

# Reference: HoVerNet Pytorch implementation https://github.com/vqdang/hover_net/blob/master/
#import cv2
#import numpy as np

#from scipy.ndimage import filters, measurements
#from scipy.ndimage.morphology import (
#    binary_dilation,
#    binary_fill_holes,
#    distance_transform_cdt,
#    distance_transform_edt,
#)

#from skimage.segmentation import watershed
#from HoVerNet_post_proc.misc.utils import get_bounding_box, remove_small_objects

#import warnings


class XentropyLoss(_Loss):
    """Cross entropy loss"""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(size_average=None, reduce=None, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Assumes NCHW shape of array, must be torch.float32 dtype

        Args:
            input (torch.Tensor): Ground truth array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes
            target (torch.Tensor): Prediction array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes

        Returns:
            torch.Tensor: Cross entropy loss, with shape () [scalar], grad_fn = MeanBackward0
        """
        # reshape
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)

        epsilon = 10e-8
        # scale preds so that the class probs of each sample sum to 1
        pred = input / torch.sum(input, -1, keepdim=True)
        # manual computation of crossentropy
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
        loss = -torch.sum((target * torch.log(pred)), -1, keepdim=True)
        loss = loss.mean() if self.reduction == "mean" else loss.sum()

        return loss


class DiceLoss(_Loss):
    """Dice loss

    Args:
        smooth (float, optional): Smoothing value. Defaults to 1e-3.
    """

    def __init__(self, smooth: float = 1e-3) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Assumes NCHW shape of array, must be torch.float32 dtype

        `pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC.

        Args:
            input (torch.Tensor): Prediction array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes
            target (torch.Tensor): Ground truth array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes

        Returns:
            torch.Tensor: Dice loss, with shape () [scalar], grad_fn=SumBackward0
        """
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        inse = torch.sum(input * target, (0, 1, 2))
        l = torch.sum(input, (0, 1, 2))
        r = torch.sum(target, (0, 1, 2))
        loss = 1.0 - (2.0 * inse + self.smooth) / (l + r + self.smooth)
        loss = torch.sum(loss)

        return loss


class MSELossMaps(_Loss):
    """Calculate mean squared error loss for combined horizontal and vertical maps of segmentation tasks."""

    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss calculation

        Args:
            input (torch.Tensor): Prediction of combined horizontal and vertical maps
                with shape (N, 2, H, W), channel 0 is vertical and channel 1 is horizontal
            target (torch.Tensor): Ground truth of combined horizontal and vertical maps
                with shape (N, 2, H, W), channel 0 is vertical and channel 1 is horizontal

        Returns:
            torch.Tensor: Mean squared error per pixel with shape (N, 2, H, W), grad_fn=SubBackward0

        """
        # reshape
        loss = input - target
        loss = (loss * loss).mean()
        return loss


class MSGELossMaps(_Loss):
    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")

    def get_sobel_kernel(
        self, size: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sobel kernel with a given size.

        Args:
            size (int): Kernel site
            device (str): Cuda device

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Horizontal and vertical sobel kernel, each with shape (size, size)
        """
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range, indexing="ij")
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    def get_gradient_hv(self, hv: torch.Tensor, device: str) -> torch.Tensor:
        """For calculating gradient of horizontal and vertical prediction map


        Args:
            hv (torch.Tensor): horizontal and vertical map
            device (str): CUDA device

        Returns:
            torch.Tensor: Gradient with same shape as input
        """
        kernel_h, kernel_v = self.get_sobel_kernel(5, device=device)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        focus: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """MSGE (Gradient of MSE) loss

        Args:
            input (torch.Tensor): Input with shape (B, C, H, W)
            target (torch.Tensor): Target with shape (B, C, H, W)
            focus (torch.Tensor): Focus, type of masking (B, C, W, W)
            device (str): CUDA device to work with.

        Returns:
            torch.Tensor: MSGE loss
        """
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        focus = focus.permute(0, 2, 3, 1)
        focus = focus[..., 1]

        focus = (focus[..., None]).float()  # assume input NHW
        focus = torch.cat([focus, focus], axis=-1).to(device)
        true_grad = self.get_gradient_hv(target, device)
        pred_grad = self.get_gradient_hv(input, device)
        loss = pred_grad - true_grad
        loss = focus * (loss * loss)
        # artificial reduce_mean with focused region
        loss = loss.sum() / (focus.sum() + 1.0e-8)
        return loss


class FocalTverskyLoss(nn.Module):
    """FocalTverskyLoss

    PyTorch implementation of the Focal Tversky Loss Function for multiple classes
    doi: 10.1109/ISBI.2019.8759329
    Abraham, N., & Khan, N. M. (2019).
    A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation.
    In International Symposium on Biomedical Imaging. https://doi.org/10.1109/isbi.2019.8759329

    @ Fabian Hörst, fabian.hoerst@uk-essen.de
    Institute for Artifical Intelligence in Medicine,
    University Medicine Essen

    Args:
        alpha_t (float, optional): Alpha parameter for tversky loss (multiplied with false-negatives). Defaults to 0.7.
        beta_t (float, optional): Beta parameter for tversky loss (multiplied with false-positives). Defaults to 0.3.
        gamma_f (float, optional): Gamma Focal parameter. Defaults to 4/3.
        smooth (float, optional): Smooting factor. Defaults to 0.000001.
    """

    def __init__(
        self,
        alpha_t: float = 0.7,
        beta_t: float = 0.3,
        gamma_f: float = 4 / 3,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.gamma_f = gamma_f
        self.smooth = smooth
        self.num_classes = 2

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss calculation

        Args:
            input (torch.Tensor): Predictions, logits (without Softmax). Shape: (B, C, H, W)
            target (torch.Tensor): Targets, either flattened (Shape: (C, H, W) or as one-hot encoded (Shape: (batch-size, C, H, W)).

        Raises:
            ValueError: Error if there is a shape missmatch

        Returns:
            torch.Tensor: FocalTverskyLoss (weighted)
        """
        input = input.permute(0, 2, 3, 1)
        if input.shape[-1] != self.num_classes:
            raise ValueError(
                "Predictions must be a logit tensor with the last dimension shape beeing equal to the number of classes"
            )
        if len(target.shape) != len(input.shape):
            # convert the targets to onehot
            target = F.one_hot(target, num_classes=self.num_classes)

        # flatten
        target = target.permute(0, 2, 3, 1)
        target = target.view(-1)
        input = torch.softmax(input, dim=-1).view(-1)

        # calculate true positives, false positives and false negatives
        tp = (input * target).sum()
        fp = ((1 - target) * input).sum()
        fn = (target * (1 - input)).sum()

        Tversky = (tp + self.smooth) / (
            tp + self.alpha_t * fn + self.beta_t * fp + self.smooth
        )
        FocalTversky = (1 - Tversky) ** self.gamma_f

        return FocalTversky


class MCFocalTverskyLoss(FocalTverskyLoss):
    """Multiclass FocalTverskyLoss

    PyTorch implementation of the Focal Tversky Loss Function for multiple classes
    doi: 10.1109/ISBI.2019.8759329
    Abraham, N., & Khan, N. M. (2019).
    A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation.
    In International Symposium on Biomedical Imaging. https://doi.org/10.1109/isbi.2019.8759329

    @ Fabian Hörst, fabian.hoerst@uk-essen.de
    Institute for Artifical Intelligence in Medicine,
    University Medicine Essen

    Args:
        alpha_t (float, optional): Alpha parameter for tversky loss (multiplied with false-negatives). Defaults to 0.7.
        beta_t (float, optional): Beta parameter for tversky loss (multiplied with false-positives). Defaults to 0.3.
        gamma_f (float, optional): Gamma Focal parameter. Defaults to 4/3.
        smooth (float, optional): Smooting factor. Defaults to 0.000001.
        num_classes (int, optional): Number of output classes. For binary segmentation, prefer FocalTverskyLoss (speed optimized). Defaults to 2.
        class_weights (List[int], optional): Weights for each class. If not provided, equal weight. Length must be equal to num_classes. Defaults to None.
    """

    def __init__(
        self,
        alpha_t: float = 0.7,
        beta_t: float = 0.3,
        gamma_f: float = 4 / 3,
        smooth: float = 0.000001,
        num_classes: int = 2,
        class_weights: List[int] = None,
    ) -> None:
        super().__init__(alpha_t, beta_t, gamma_f, smooth)
        self.num_classes = num_classes
        if class_weights is None:
            self.class_weights = [1 for i in range(self.num_classes)]
        else:
            assert (
                len(class_weights) == self.num_classes
            ), "Please provide matching weights"
            self.class_weights = class_weights
        self.class_weights = torch.Tensor(self.class_weights)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss calculation

        Args:
            input (torch.Tensor): Predictions, logits (without Softmax). Shape: (B, num_classes, H, W)
            target (torch.Tensor): Targets, either flattened (Shape: (B, H, W) or as one-hot encoded (Shape: (B, num_classes, H, W)).

        Raises:
            ValueError: Error if there is a shape missmatch

        Returns:
            torch.Tensor: FocalTverskyLoss (weighted)
        """
        input = input.permute(0, 2, 3, 1)
        if input.shape[-1] != self.num_classes:
            raise ValueError(
                "Predictions must be a logit tensor with the last dimension shape beeing equal to the number of classes"
            )
        if len(target.shape) != len(input.shape):
            # convert the targets to onehot
            target = F.one_hot(target, num_classes=self.num_classes)

        target = target.permute(0, 2, 3, 1)
        # Softmax
        input = torch.softmax(input, dim=-1)

        # Reshape
        input = torch.permute(input, (3, 1, 2, 0))
        target = torch.permute(target, (3, 1, 2, 0))

        input = torch.flatten(input, start_dim=1)
        target = torch.flatten(target, start_dim=1)

        tp = torch.sum(input * target, 1)
        fp = torch.sum((1 - target) * input, 1)
        fn = torch.sum(target * (1 - input), 1)

        Tversky = (tp + self.smooth) / (
            tp + self.alpha_t * fn + self.beta_t * fp + self.smooth
        )
        FocalTversky = (1 - Tversky) ** self.gamma_f

        self.class_weights = self.class_weights.to(FocalTversky.device)
        return torch.sum(self.class_weights * FocalTversky)


class WeightedBaseLoss(nn.Module):
    """Init a base class for weighted cross entropy based losses.

    Enables weighting for object instance edges and classes.

    Adapted/Copied from: https://github.com/okunator/cellseg_models.pytorch (10.5281/zenodo.7064617)

    Args:
        apply_sd (bool, optional): If True, Spectral decoupling regularization will be applied to the
            loss matrix. Defaults to False.
        apply_ls (bool, optional): If True, Label smoothing will be applied to the target.. Defaults to False.
        apply_svls (bool, optional): If True, spatially varying label smoothing will be applied to the target. Defaults to False.
        apply_mask (bool, optional): If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W). Defaults to False.
        class_weights (torch.Tensor, optional): Class weights. A tensor of shape (C, ). Defaults to None.
        edge_weight (float, optional): Weight for the object instance border pixels. Defaults to None.
    """

    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        class_weights: torch.Tensor = None,
        edge_weight: float = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.apply_sd = apply_sd
        self.apply_ls = apply_ls
        self.apply_svls = apply_svls
        self.apply_mask = apply_mask
        self.class_weights = class_weights
        self.edge_weight = edge_weight

    def apply_spectral_decouple(
        self, loss_matrix: torch.Tensor, yhat: torch.Tensor, lam: float = 0.01
    ) -> torch.Tensor:
        """Apply spectral decoupling L2 norm after the loss.

        https://arxiv.org/abs/2011.09468

        Args:
            loss_matrix (torch.Tensor): Pixelwise losses. A tensor of shape (B, H, W).
            yhat (torch.Tensor): The pixel predictions of the model. Shape (B, C, H, W).
            lam (float, optional): Lambda constant.. Defaults to 0.01.

        Returns:
            torch.Tensor: SD-regularized loss matrix. Same shape as input.
        """
        return loss_matrix + (lam / 2) * (yhat**2).mean(axis=1)

    def apply_ls_to_target(
        self,
        target: torch.Tensor,
        num_classes: int,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        """_summary_

        Args:
            target (torch.Tensor): Number of classes in the data.
            num_classes (int): The target one hot tensor. Shape (B, C, H, W)
            label_smoothing (float, optional):  The smoothing coeff alpha. Defaults to 0.1.

        Returns:
            torch.Tensor: Label smoothed target. Same shape as input.
        """
        return target * (1 - label_smoothing) + label_smoothing / num_classes

    def apply_svls_to_target(
        self,
        target: torch.Tensor,
        num_classes: int,
        kernel_size: int = 5,
        sigma: int = 3,
        **kwargs,
    ) -> torch.Tensor:
        """Apply spatially varying label smoothihng to target map.

        https://arxiv.org/abs/2104.05788

        Args:
            target (torch.Tensor): The target one hot tensor. Shape (B, C, H, W).
            num_classes (int):  Number of classes in the data.
            kernel_size (int, optional): Size of a square kernel.. Defaults to 5.
            sigma (int, optional): The std of the gaussian. Defaults to 3.

        Returns:
            torch.Tensor: Label smoothed target. Same shape as input.
        """
        my, mx = kernel_size // 2, kernel_size // 2
        gaussian_kernel = gaussian_kernel2d(
            kernel_size, sigma, num_classes, device=target.device
        )
        neighborsum = (1 - gaussian_kernel[..., my, mx]) + 1e-16
        gaussian_kernel = gaussian_kernel.clone()
        gaussian_kernel[..., my, mx] = neighborsum
        svls_kernel = gaussian_kernel / neighborsum[0]

        return filter2D(target.float(), svls_kernel) / svls_kernel[0].sum()

    def apply_class_weights(
        self, loss_matrix: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Multiply pixelwise loss matrix by the class weights.

        NOTE: No normalization

        Args:
            loss_matrix (torch.Tensor): Pixelwise losses. A tensor of shape (B, H, W).
            target (torch.Tensor): The target mask. Shape (B, H, W).

        Returns:
            torch.Tensor: The loss matrix scaled with the weight matrix. Shape (B, H, W).
        """
        weight_mat = self.class_weights[target.long()].to(target.device)  # to (B, H, W)
        loss = loss_matrix * weight_mat

        return loss

    def apply_edge_weights(
        self, loss_matrix: torch.Tensor, weight_map: torch.Tensor
    ) -> torch.Tensor:
        """Apply weights to the object boundaries.

        Basically just computes `edge_weight`**`weight_map`.

        Args:
            loss_matrix (torch.Tensor): Pixelwise losses. A tensor of shape (B, H, W).
            weight_map (torch.Tensor): Map that points to the pixels that will be weighted. Shape (B, H, W).

        Returns:
            torch.Tensor: The loss matrix scaled with the nuclear boundary weights. Shape (B, H, W).
        """
        return loss_matrix * self.edge_weight**weight_map

    def apply_mask_weight(
        self, loss_matrix: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        """Apply a mask to the loss matrix.

        Args:
            loss_matrix (torch.Tensor): Pixelwise losses. A tensor of shape (B, H, W).
            mask (torch.Tensor): The mask. Shape (B, H, W).
            norm (bool, optional): If True, the loss matrix will be normalized by the mean of the mask. Defaults to True.

        Returns:
            torch.Tensor: The loss matrix scaled with the mask. Shape (B, H, W).
        """
        loss_matrix *= mask
        if norm:
            norm_mask = torch.mean(mask.float()) + 1e-7
            loss_matrix /= norm_mask

        return loss_matrix

    def extra_repr(self) -> str:
        """Add info to print."""
        s = "apply_sd={apply_sd}, apply_ls={apply_ls}, apply_svls={apply_svls}, apply_mask={apply_mask}, class_weights={class_weights}, edge_weight={edge_weight}"  # noqa
        return s.format(**self.__dict__)


class MAEWeighted(WeightedBaseLoss):
    """Compute the MAE loss. Used in the stardist method.

    Stardist:
    https://arxiv.org/pdf/1806.03535.pdf
    Adapted/Copied from: https://github.com/okunator/cellseg_models.pytorch (10.5281/zenodo.7064617)

    NOTE: We have added the option to apply spectral decoupling and edge weights
    to the loss matrix.

    Args:
        alpha (float, optional): Weight regulizer b/w [0,1]. In stardist repo, this is the
        'train_background_reg' parameter. Defaults to 1e-4.
        apply_sd (bool, optional): If True, Spectral decoupling regularization will be applied  to the
        loss matrix. Defaults to False.
        apply_mask (bool, optional): f True, a mask will be applied to the loss matrix. Mask shape: (B, H, W). Defaults to False.
        edge_weight (float, optional): Weight that is added to object borders. Defaults to None.
    """

    def __init__(
        self,
        alpha: float = 1e-4,
        apply_sd: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        **kwargs,
    ) -> None:
        super().__init__(apply_sd, False, False, apply_mask, False, edge_weight)
        self.alpha = alpha
        self.eps = 1e-7

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the masked MAE loss.

        Args:
            input (torch.Tensor): The prediction map. Shape (B, C, H, W).
            target (torch.Tensor): The ground truth annotations. Shape (B, H, W).
            target_weight (torch.Tensor, optional): The edge weight map. Shape (B, H, W). Defaults to None.
            mask (torch.Tensor, optional): The mask map. Shape (B, H, W). Defaults to None.

        Raises:
            ValueError: Pred and target shapes must match.

        Returns:
            torch.Tensor: Computed MAE loss (scalar).
        """
        yhat = input
        n_classes = yhat.shape[1]
        if target.size() != yhat.size():
            target = target.unsqueeze(1).repeat_interleave(n_classes, dim=1)

        if not yhat.shape == target.shape:
            raise ValueError(
                f"Pred and target shapes must match. Got: {yhat.shape}, {target.shape}"
            )

        # compute the MAE loss with alpha as weight
        mae_loss = torch.mean(torch.abs(target - yhat), axis=1)  # (B, H, W)

        if self.apply_mask and mask is not None:
            mae_loss = self.apply_mask_weight(mae_loss, mask, norm=True)  # (B, H, W)

            # add the background regularization
            if self.alpha > 0:
                reg = torch.mean(((1 - mask).unsqueeze(1)) * torch.abs(yhat), axis=1)
                mae_loss += self.alpha * reg

        if self.apply_sd:
            mae_loss = self.apply_spectral_decouple(mae_loss, yhat)

        if self.edge_weight is not None:
            mae_loss = self.apply_edge_weights(mae_loss, target_weight)

        return mae_loss.mean()


class MSEWeighted(WeightedBaseLoss):
    """MSE-loss.

    Args:
        apply_sd (bool, optional): If True, Spectral decoupling regularization will be applied  to the
            loss matrix. Defaults to False.
        apply_ls (bool, optional): If True, Label smoothing will be applied to the target. Defaults to False.
        apply_svls (bool, optional): If True, spatially varying label smoothing will be applied to the target. Defaults to False.
        apply_mask (bool, optional): If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W). Defaults to False.
        edge_weight (float, optional): Weight that is added to object borders. Defaults to None.
        class_weights (torch.Tensor, optional): Class weights. A tensor of shape (n_classes,). Defaults to None.
    """

    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(
            apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
        )

    @staticmethod
    def tensor_one_hot(type_map: torch.Tensor, n_classes: int) -> torch.Tensor:
        """Convert a segmentation mask into one-hot-format.

        I.e. Takes in a segmentation mask of shape (B, H, W) and reshapes it
        into a tensor of shape (B, C, H, W).

        Args:
            type_map (torch.Tensor):  Multi-label Segmentation mask. Shape (B, H, W).
            n_classes (int): Number of classes. (Zero-class included.)

        Raises:
            TypeError: Input `type_map` should have dtype: torch.int64.

        Returns:
            torch.Tensor: A one hot tensor. Shape: (B, C, H, W). Dtype: torch.FloatTensor.
        """
        if not type_map.dtype == torch.int64:
            raise TypeError(
                f"""
                Input `type_map` should have dtype: torch.int64. Got: {type_map.dtype}."""
            )

        one_hot = torch.zeros(
            type_map.shape[0],
            n_classes,
            *type_map.shape[1:],
            device=type_map.device,
            dtype=type_map.dtype,
        )

        return one_hot.scatter_(dim=1, index=type_map.unsqueeze(1), value=1.0) + 1e-7

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the MSE-loss.

        Args:
            input (torch.Tensor): The prediction map. Shape (B, C, H, W, C).
            target (torch.Tensor): The ground truth annotations. Shape (B, H, W).
            target_weight (torch.Tensor, optional):  The edge weight map. Shape (B, H, W). Defaults to None.
            mask (torch.Tensor, optional): The mask map. Shape (B, H, W). Defaults to None.

        Returns:
            torch.Tensor: Computed MSE loss (scalar).
        """
        yhat = input
        target_one_hot = target
        num_classes = yhat.shape[1]

        if target.size() != yhat.size():
            if target.dtype == torch.float32:
                target_one_hot = target.unsqueeze(1)
            else:
                target_one_hot = MSEWeighted.tensor_one_hot(target, num_classes)

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        mse = F.mse_loss(yhat, target_one_hot, reduction="none")  # (B, C, H, W)
        mse = torch.mean(mse, dim=1)  # to (B, H, W)

        if self.apply_mask and mask is not None:
            mse = self.apply_mask_weight(mse, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            mse = self.apply_spectral_decouple(mse, yhat)

        if self.class_weights is not None:
            mse = self.apply_class_weights(mse, target)

        if self.edge_weight is not None:
            mse = self.apply_edge_weights(mse, target_weight)

        return torch.mean(mse)


class BCEWeighted(WeightedBaseLoss):
    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        """Binary cross entropy loss with weighting and other tricks.

        Parameters
        ----------
        apply_sd : bool, default=False
            If True, Spectral decoupling regularization will be applied  to the
            loss matrix.
        apply_ls : bool, default=False
            If True, Label smoothing will be applied to the target.
        apply_svls : bool, default=False
            If True, spatially varying label smoothing will be applied to the target
        apply_mask : bool, default=False
            If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W)
        edge_weight : float, default=None
            Weight that is added to object borders.
        class_weights : torch.Tensor, default=None
            Class weights. A tensor of shape (n_classes,).
        """
        super().__init__(
            apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
        )
        self.eps = 1e-8

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute binary cross entropy loss.

        Parameters
        ----------
            yhat : torch.Tensor
                The prediction map. Shape (B, C, H, W).
            target : torch.Tensor
                the ground truth annotations. Shape (B, H, W).
            target_weight : torch.Tensor, default=None
                The edge weight map. Shape (B, H, W).
            mask : torch.Tensor, default=None
                The mask map. Shape (B, H, W).

        Returns
        -------
            torch.Tensor:
                Computed BCE loss (scalar).
        """
        # Logits input
        yhat = input
        num_classes = yhat.shape[1]
        yhat = torch.clip(yhat, self.eps, 1.0 - self.eps)

        if target.size() != yhat.size():
            target = target.unsqueeze(1).repeat_interleave(num_classes, dim=1)

        if self.apply_svls:
            target = self.apply_svls_to_target(target, num_classes, **kwargs)

        if self.apply_ls:
            target = self.apply_ls_to_target(target, num_classes, **kwargs)

        bce = F.binary_cross_entropy_with_logits(
            yhat.float(), target.float(), reduction="none"
        )  # (B, C, H, W)
        bce = torch.mean(bce, dim=1)  # (B, H, W)

        if self.apply_mask and mask is not None:
            bce = self.apply_mask_weight(bce, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            bce = self.apply_spectral_decouple(bce, yhat)

        if self.class_weights is not None:
            bce = self.apply_class_weights(bce, target)

        if self.edge_weight is not None:
            bce = self.apply_edge_weights(bce, target_weight)

        return torch.mean(bce)


# class BCEWeighted(WeightedBaseLoss):
#     """Binary cross entropy loss with weighting and other tricks.
#     Adapted/Copied from: https://github.com/okunator/cellseg_models.pytorch (10.5281/zenodo.7064617)

#     Args:
#         apply_sd (bool, optional): If True, Spectral decoupling regularization will be applied  to the
#             loss matrix. Defaults to False.
#         apply_ls (bool, optional): If True, Label smoothing will be applied to the target. Defaults to False.
#         apply_svls (bool, optional): If True, spatially varying label smoothing will be applied to the target. Defaults to False.
#         apply_mask (bool, optional): If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W). Defaults to False.
#         edge_weight (float, optional):  Weight that is added to object borders. Defaults to None.
#         class_weights (torch.Tensor, optional): Class weights. A tensor of shape (n_classes,). Defaults to None.
#     """

#     def __init__(
#         self,
#         apply_sd: bool = False,
#         apply_ls: bool = False,
#         apply_svls: bool = False,
#         apply_mask: bool = False,
#         edge_weight: float = None,
#         class_weights: torch.Tensor = None,
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
#         )
#         self.eps = 1e-8

#     def forward(
#         self,
#         input: torch.Tensor,
#         target: torch.Tensor,
#         target_weight: torch.Tensor = None,
#         mask: torch.Tensor = None,
#         **kwargs,
#     ) -> torch.Tensor:
#         """Compute binary cross entropy loss.

#         Args:
#             input (torch.Tensor): The prediction map. We internally convert back via logit function. Shape (B, C, H, W).
#             target (torch.Tensor): the ground truth annotations. Shape (B, H, W).
#             target_weight (torch.Tensor, optional): The edge weight map. Shape (B, H, W). Defaults to None.
#             mask (torch.Tensor, optional): The mask map. Shape (B, H, W). Defaults to None.

#         Returns:
#             torch.Tensor: Computed BCE loss (scalar).
#         """
#         yhat = input
#         yhat = torch.special.logit(yhat)
#         num_classes = yhat.shape[1]
#         yhat = torch.clip(yhat, self.eps, 1.0 - self.eps)

#         if target.size() != yhat.size():
#             target = target.unsqueeze(1).repeat_interleave(num_classes, dim=1)

#         if self.apply_svls:
#             target = self.apply_svls_to_target(target, num_classes, **kwargs)

#         if self.apply_ls:
#             target = self.apply_ls_to_target(target, num_classes, **kwargs)

#         bce = F.binary_cross_entropy_with_logits(
#             yhat.float(), target.float(), reduction="none"
#         )  # (B, C, H, W)
#         bce = torch.mean(bce, dim=1)  # (B, H, W)

#         if self.apply_mask and mask is not None:
#             bce = self.apply_mask_weight(bce, mask, norm=False)  # (B, H, W)

#         if self.apply_sd:
#             bce = self.apply_spectral_decouple(bce, yhat)

#         if self.class_weights is not None:
#             bce = self.apply_class_weights(bce, target)

#         if self.edge_weight is not None:
#             bce = self.apply_edge_weights(bce, target_weight)

#         return torch.mean(bce)


class CEWeighted(WeightedBaseLoss):
    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        """Cross-Entropy loss with weighting.

        Parameters
        ----------
        apply_sd : bool, default=False
            If True, Spectral decoupling regularization will be applied  to the
            loss matrix.
        apply_ls : bool, default=False
            If True, Label smoothing will be applied to the target.
        apply_svls : bool, default=False
            If True, spatially varying label smoothing will be applied to the target
        apply_mask : bool, default=False
            If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W)
        edge_weight : float, default=None
            Weight that is added to object borders.
        class_weights : torch.Tensor, default=None
            Class weights. A tensor of shape (n_classes,).
        """
        super().__init__(
            apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
        )
        self.eps = 1e-8

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the cross entropy loss.

        Parameters
        ----------
            yhat : torch.Tensor
                The prediction map. Shape (B, C, H, W).
            target : torch.Tensor
                the ground truth annotations. Shape (B, H, W).
            target_weight : torch.Tensor, default=None
                The edge weight map. Shape (B, H, W).
            mask : torch.Tensor, default=None
                The mask map. Shape (B, H, W).

        Returns
        -------
            torch.Tensor:
                Computed CE loss (scalar).
        """
        yhat = input  # TODO: remove doubled Softmax -> this function needs logits instead of softmax output
        input_soft = F.softmax(yhat, dim=1) + self.eps  # (B, C, H, W)
        num_classes = yhat.shape[1]
        if len(target.shape) != len(yhat.shape) and target.shape[1] != num_classes:
            target_one_hot = MSEWeighted.tensor_one_hot(
                target, num_classes
            )  # (B, C, H, W)
        else:
            target_one_hot = target
            target = torch.argmax(target, dim=1)
        assert target_one_hot.shape == yhat.shape

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        loss = -torch.sum(target_one_hot * torch.log(input_soft), dim=1)  # (B, H, W)

        if self.apply_mask and mask is not None:
            loss = self.apply_mask_weight(loss, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            loss = self.apply_spectral_decouple(loss, yhat)

        if self.class_weights is not None:
            loss = self.apply_class_weights(loss, target)

        if self.edge_weight is not None:
            loss = self.apply_edge_weights(loss, target_weight)

        return loss.mean()


# class CEWeighted(WeightedBaseLoss):
#     """Cross-Entropy loss with weighting.
#     Adapted/Copied from: https://github.com/okunator/cellseg_models.pytorch (10.5281/zenodo.7064617)

#     Args:
#         apply_sd (bool, optional): If True, Spectral decoupling regularization will be applied to the loss matrix. Defaults to False.
#         apply_ls (bool, optional): If True, Label smoothing will be applied to the target. Defaults to False.
#         apply_svls (bool, optional): If True, spatially varying label smoothing will be applied to the target. Defaults to False.
#         apply_mask (bool, optional): If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W). Defaults to False.
#         edge_weight (float, optional): Weight that is added to object borders. Defaults to None.
#         class_weights (torch.Tensor, optional): Class weights. A tensor of shape (n_classes,). Defaults to None.
#         logits (bool, optional): If work on logit values. Defaults to False. Defaults to False.
#     """

#     def __init__(
#         self,
#         apply_sd: bool = False,
#         apply_ls: bool = False,
#         apply_svls: bool = False,
#         apply_mask: bool = False,
#         edge_weight: float = None,
#         class_weights: torch.Tensor = None,
#         logits: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
#         )
#         self.eps = 1e-8
#         self.logits = logits

#     def forward(
#         self,
#         input: torch.Tensor,
#         target: torch.Tensor,
#         target_weight: torch.Tensor = None,
#         mask: torch.Tensor = None,
#         **kwargs,
#     ) -> torch.Tensor:
#         """Compute the cross entropy loss.

#         Args:
#             input (torch.Tensor): The prediction map. Shape (B, C, H, W).
#             target (torch.Tensor): The ground truth annotations. Shape (B, H, W).
#             target_weight (torch.Tensor, optional): The edge weight map. Shape (B, H, W). Defaults to None.
#             mask (torch.Tensor, optional): The mask map. Shape (B, H, W). Defaults to None.

#         Returns:
#             torch.Tensor: Computed CE loss (scalar).
#         """
#         yhat = input
#         if self.logits:
#             input_soft = (
#                 F.softmax(yhat, dim=1) + self.eps
#             )  # (B, C, H, W) # check if doubled softmax
#         else:
#             input_soft = input

#         num_classes = yhat.shape[1]
#         if len(target.shape) != len(yhat.shape) and target.shape[1] != num_classes:
#             target_one_hot = MSEWeighted.tensor_one_hot(
#                 target, num_classes
#             )  # (B, C, H, W)
#         else:
#             target_one_hot = target
#             target = torch.argmax(target, dim=1)
#         assert target_one_hot.shape == yhat.shape

#         if self.apply_svls:
#             target_one_hot = self.apply_svls_to_target(
#                 target_one_hot, num_classes, **kwargs
#             )

#         if self.apply_ls:
#             target_one_hot = self.apply_ls_to_target(
#                 target_one_hot, num_classes, **kwargs
#             )

#         loss = -torch.sum(target_one_hot * torch.log(input_soft), dim=1)  # (B, H, W)

#         if self.apply_mask and mask is not None:
#             loss = self.apply_mask_weight(loss, mask, norm=False)  # (B, H, W)

#         if self.apply_sd:
#             loss = self.apply_spectral_decouple(loss, yhat)

#         if self.class_weights is not None:
#             loss = self.apply_class_weights(loss, target)

#         if self.edge_weight is not None:
#             loss = self.apply_edge_weights(loss, target_weight)

#         return loss.mean()


### Stardist loss functions
class L1LossWeighted(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        l1loss = F.l1_loss(input, target, size_average=True, reduce=False)
        l1loss = torch.mean(l1loss, dim=1)
        if target_weight is not None:
            l1loss = torch.mean(target_weight * l1loss)
        else:
            l1loss = torch.mean(l1loss)
        return l1loss

#class BendingLossCalculator:
#    def __init__(self, alpha):
#        self.alpha = alpha

#    def calculate_curvature(self, point, prev_point, next_point):
#        v1 = point - prev_point
#        v2 = next_point - point
#        cross_product = np.cross(v1, v2)
#        dot_product = np.dot(v1, v2)
#        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
#        curvature = 2 * np.abs(cross_product) / (norm_product + dot_product)
#        return curvature

#    def calculate_bending_energy(self, curvature, v1, v2):
#        bending_energy = curvature ** 2 / (np.linalg.norm(v1) + np.linalg.norm(v2))
#        return bending_energy

#    def calculate_bending_loss(self, contour_points):
#        m = len(contour_points)
#        total_bending_energy = 0

#        for i in range(1, m - 1):
#            prev_point = contour_points[i - 1]
#            point = contour_points[i]
#            next_point = contour_points[i + 1]
#            curvature = self.calculate_curvature(point, prev_point, next_point)
#            v1 = point - prev_point
#            v2 = next_point - point
#            bending_energy = self.calculate_bending_energy(curvature, v1, v2)
#            total_bending_energy += bending_energy

#        bending_loss = total_bending_energy / m
#        return bending_loss

    #def calculate_total_loss(self, segmentation_loss, contour_points):
    #    bending_loss = self.calculate_bending_loss(contour_points)
    #    total_loss = segmentation_loss + self.alpha * bending_loss
    #    return total_loss
    
#    def forward(
#        self,
#        input: torch.Tensor,
#        target: torch.Tensor = None,
#        target_weight: torch.Tensor = None,
#    ) -> torch.Tensor:
#        l1loss = F.l1_loss(input, target, size_average=True, reduce=False)
#        l1loss = torch.mean(l1loss, dim=1)
#        if target_weight is not None:
#            l1loss = torch.mean(target_weight * l1loss)
#        else:
#            l1loss = torch.mean(l1loss)
#        return l1loss
    
"""
class BendingLoss(nn.Module):
    def __init__(self,alpha) -> None:
        super().__init__()
        self.alpha = alpha
    
    def calculate_curvature(self, point, prev_point, next_point):
        v1 = point - prev_point
        v2 = next_point - point
        cross_product = np.cross(v1, v2)
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if cross_product==0:
            curvature = 0
        else:
            curvature = 2 * np.abs(cross_product) / (norm_product + dot_product)
        #if np.isnan(curvature):
        #    print(f"v1 = {v1} | v2 = {v2}")
        #    print(f"cross_product = {cross_product}")
        #    print(f"cross_product = {cross_product} | dot_product = {dot_product} | norm_product = {norm_product}")
        #    print(f"curvature = {curvature}")
        return curvature
    
    def calculate_bending_energy(self, curvature, v1, v2):
        bending_energy = curvature ** 2 / (np.linalg.norm(v1) + np.linalg.norm(v2))
        return bending_energy
    
    def calculate_bending_loss(self, contour_points):
        m = len(contour_points)
        total_bending_energy = 0

        for i in range(1, m - 1):
            prev_point = contour_points[i - 1]
            point = contour_points[i]
            next_point = contour_points[i + 1]
            curvature = self.calculate_curvature(point, prev_point, next_point)
            #break
            v1 = point - prev_point
            v2 = next_point - point
            bending_energy = self.calculate_bending_energy(curvature, v1, v2)
            total_bending_energy += bending_energy

        #bending_loss = total_bending_energy / m
        #return bending_loss
        return total_bending_energy
    
    def forward(
        self,
        input: torch.Tensor,
        #target: torch.Tensor = None,
        #target_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        #inst_info_dict = input
        #print(len(inst_info_dict))
        #print(len(inst_info_dict[0]))
        #print(inst_info_dict[0][0])
        bending_losses = [] # list of bending energies of all images in a batch
        for inst_info_dict in input: # iterate through a batch of images
            bending_energies = []
            for inst_id in list(inst_info_dict.keys()): # iterate through all nuclei instances in a single image
                b = self.calculate_bending_loss(inst_info_dict[inst_id]["contour"]) # bending energy of a single nucleus instance
                #break
                bending_energies.append(b)
            #bending_energies = torch.Tensor(bending_energies, dtype=torch.float32) # List of bending energies of all instances of nuclei
            if len(bending_energies)>0:
                bending_losses.append(sum(bending_energies)/len(bending_energies)) # Mean of bending energes of all contour points of nuclei in an image
            else:
                bending_losses.append(0)
            #break
        #bending_losses = torch.Tensor(bending_losses, dtype=torch.float32)
        #bending_loss = torch.mean(bending_losses,dim=1)
        bending_loss = sum(bending_losses)/len(bending_losses)
        
        return bending_loss,self.alpha
"""
# Bending Loss Pytorch Custom
class BendingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = 1
        self.mu = 0.75 # Change
        print("---------------------Bending Loss----------------------------")
    
    def get_mask(self,tensor):
        """
        Classifies pixels based on their probabilities of belonging to the nucleus. (Same as classify_nucleus() from SAMSLoss)

        Args:
        tensor (torch.Tensor): A tensor of shape (2, 256, 256) where the first channel
                            contains the probabilities of a pixel belonging to the background
                            and the second channel contains the probabilities of a pixel 
                            belonging to the nucleus.

        Returns:
        torch.Tensor: A binary tensor of shape (256, 256) where an element is 1 if the pixel
                    is more likely to belong to the nucleus (probability > 0.5) and 0 otherwise.
        """
        # Ensure the input tensor is of the correct shape
        assert tensor.shape == (2, 256, 256), "Input tensor must be of shape (2, 256, 256)"
        
        # Get the probability of a pixel belonging to the nucleus
        nucleus_prob = tensor[1]
        
        # Create the binary output tensor
        #output = (nucleus_prob > 0.5).int()
        #output = torch.sigmoid(nucleus_prob - 0.5).int()
        output = torch.where(nucleus_prob>0.5,1,0).int()
        #print(torch.nonzero(output == 1, as_tuple=False))
        #print(output.requires_grad)

        return output
    
    def get_contour_points(self,binary_mask):
        # Define 8-connectivity kernel
        kernel = torch.tensor([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]], dtype=binary_mask.dtype).unsqueeze(0).unsqueeze(0).cpu()#.to(binary_mask.device)
        
        # Compute the number of foreground neighbors for each pixel
        #print(kernel.dtype)
        #print(binary_mask.requires_grad) # binary_mask does not support backpropagagtion
        binary_mask = binary_mask.unsqueeze(0).unsqueeze(0).detach().cpu()
        #print(binary_mask.shape)
        #print(kernel.shape)
        neighbor_count = F.conv2d(binary_mask, kernel, padding=1).squeeze()
        #print(neighbor_count.dtype)
        #output = torch.nonzero(neighbor_count>0).int()
        #print(output)
        # Condition 1: Pixel value is 1 (foreground)
        condition1 = binary_mask == 1
        
        # Condition 2: Not all adjacent pixels are 1 (neighbor count less than 8)
        condition2 = neighbor_count < 8
        
        # Combine conditions to get the contour
        contour = condition1 & condition2

        contour = contour.squeeze()
        #print(contour.shape)
        
        return contour

    def calculate_delta(self, point, prev_point, next_point):
        v1 = point - prev_point
        v2 = next_point - point
        # Calculate the 2D cross product as a scalar value
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        
        # Determine if the point is concave or convex
        if cross_product >= 0:
            #return "convex"
            return 0
        else: #cross_product < 0:
            #return "concave"
            return 1
        

    def calculate_curvature(self, point, prev_point, next_point):
        v1 = (point - prev_point)#.detach().cpu().numpy()
        v2 = (next_point - point)#.detach().cpu().numpy()
        #cross_product = torch.tensor(np.cross(v1, v2)).to(point.device)
        #print(torch.tensor(np.cross(v1, v2)))
        #dot_product = torch.tensor(np.dot(v1, v2)).to(point.device)
        # Manually calculate the 2D cross product as a scalar value
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        # Manually calculate the dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        # Manually calculate the norms
        #v1 = point - prev_point
        #v2 = next_point - point
        norm_v1 = torch.sqrt(v1[0]**2 + v1[1]**2)
        norm_v2 = torch.sqrt(v2[0]**2 + v2[1]**2)
        norm_product = norm_v1 * norm_v2
        
        if cross_product.item() == 0:
            curvature = torch.tensor(0).to(point.device)
        else:
            curvature = 2 * torch.abs(cross_product) / (norm_product + dot_product)
        #print(type(curvature))
        return curvature
    
    def calculate_bending_energy(self, curvature, delta, v1, v2):
        # Manually calculate the norms
        norm_v1 = torch.sqrt(v1[0]**2 + v1[1]**2)
        norm_v2 = torch.sqrt(v2[0]**2 + v2[1]**2)
        bending_energy = (1-delta+delta*self.mu)*curvature ** 2 / (norm_v1 + norm_v2)
        #print(type(bending_energy))
        return bending_energy
    
    def calculate_bending_loss(self, contour_points):
        m = len(contour_points)
        total_bending_energy = 0

        for i in range(1, m - 1):
            prev_point = contour_points[i - 1]
            point = contour_points[i]
            next_point = contour_points[i + 1]
            curvature = self.calculate_curvature(point, prev_point, next_point)
            delta = self.calculate_delta(point, prev_point, next_point)
            #print(curvature)
            #break
            v1 = point - prev_point
            v2 = next_point - point
            bending_energy = self.calculate_bending_energy(curvature, delta, v1, v2)
            total_bending_energy += bending_energy
        #print(type(total_bending_energy))

        return total_bending_energy

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        channel 0 - background
        channel 1 - nuclei
        """
        #print(input.shape)
        #print(input.requires_grad)
        total_be = 0
        for i,pred in enumerate(target):
            #print(pred[0])
            #print(torch.nonzero(pred > 0.5, as_tuple=False))
            mask = self.get_mask(pred)
            #print(torch.nonzero(mask == 1, as_tuple=False))
            contours = self.get_contour_points(mask).to(pred[0].device)
            #save_image(contours.unsqueeze(0).float(),'contours.png')
            #print(mask.shape)
            #print(contours)
            #contours = pred[0]*contours
            #print(contours.requires_grad)
            contour_points = torch.nonzero(contours>0, as_tuple=False)
            b = self.calculate_bending_loss(contour_points)
            #print(b)
            total_be += b
            #break
        total_be = total_be/len(input)
        return total_be


        """
        bending_losses = torch.tensor([], dtype=torch.float32).to(self.device)
        for inst_info_dict in input: # iterate through a batch of images
            bending_energies = torch.tensor([], dtype=torch.float32).to(self.device)
            for inst_id in inst_info_dict.keys(): # iterate through all nuclei instances in a single image
                contour_points = inst_info_dict[inst_id]["contour"]
                contour_points = torch.tensor(contour_points, dtype=torch.float32).to(self.device)
                b = self.calculate_bending_loss(contour_points) # Convert contour points to tensor
                bending_energies = torch.cat((bending_energies, b.unsqueeze(0)))
            
            if bending_energies.numel() > 0:
                bending_losses = torch.cat((bending_losses, torch.tensor([bending_energies.mean()]).to(self.device)))
            else:
                bending_losses = torch.cat((bending_losses, torch.tensor([0.0]).to(self.device)))
        
        bending_loss = bending_losses.mean()
        return bending_loss, self.alpha
        """

# SAMS-NET Loss function
class SAMSLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reduction = None
        #print("-----------------------SAMSLoss---------------------------------------------")
    
    def classify_nucleus(self,tensor):
        """
        Classifies pixels based on their probabilities of belonging to the nucleus.

        Args:
        tensor (torch.Tensor): A tensor of shape (2, 256, 256) where the first channel
                            contains the probabilities of a pixel belonging to the background
                            and the second channel contains the probabilities of a pixel 
                            belonging to the nucleus.

        Returns:
        torch.Tensor: A binary tensor of shape (256, 256) where an element is 1 if the pixel
                    is more likely to belong to the nucleus (probability > 0.5) and 0 otherwise.
        """
        # Ensure the input tensor is of the correct shape
        assert tensor.shape == (2, 256, 256), "Input tensor must be of shape (2, 256, 256)"
        
        # Get the probability of a pixel belonging to the nucleus
        nucleus_prob = tensor[1]
        
        # Create the binary output tensor
        #output = (nucleus_prob > 0.5).int()
        #output = torch.sigmoid(nucleus_prob - 0.5) # Differentiable----but gives wrong answer
        output = torch.where(nucleus_prob>0.5,1,0).int() # Non-differentiable
        #print(torch.nonzero(output == 1, as_tuple=False))
        #print(f"mask output: {output.requires_grad}")
        #print("Bending Loss")
        #print(torch.nonzero(output == 1, as_tuple=False))
        return output

    def get_H_channel(self,img):
        """
        !! Does not support backpropagation
        Returns the Haematoxylin channel of an image
        """
        ihc_hed = rgb2hed(img.permute(1,2,0).detach().cpu().numpy())

        return ihc_hed[:, :, 0]


    """
    def chamfer_distance(self,img):
        #---------------------------------------
        #!!Convert to Pytorch!!
        #Source: https://stackoverflow.com/questions/53678520/speed-up-computation-for-distance-transform-on-image-in-python
        #----------------------------------------------
        w, h = img.shape
        dt = np.zeros((w,h), np.uint32)
        # Forward pass
        x = 0
        y = 0
        if img[x,y] == 0:
            dt[x,y] = 65535 # some large value
        for x in range(1, w):
            if img[x,y] == 0:
                dt[x,y] = 3 + dt[x-1,y]
        for y in range(1, h):
            x = 0
            if img[x,y] == 0:
                dt[x,y] = min(3 + dt[x,y-1], 4 + dt[x+1,y-1])
            for x in range(1, w-1):
                if img[x,y] == 0:
                    dt[x,y] = min(4 + dt[x-1,y-1], 3 + dt[x,y-1], 4 + dt[x+1,y-1], 3 + dt[x-1,y])
            x = w-1
            if img[x,y] == 0:
                dt[x,y] = min(4 + dt[x-1,y-1], 3 + dt[x,y-1], 3 + dt[x-1,y])
        # Backward pass
        for x in range(w-2, -1, -1):
            y = h-1
            if img[x,y] == 0:
                dt[x,y] = min(dt[x,y], 3 + dt[x+1,y])
        for y in range(h-2, -1, -1):
            x = w-1
            if img[x,y] == 0:
                dt[x,y] = min(dt[x,y], 3 + dt[x,y+1], 4 + dt[x-1,y+1])
            for x in range(1, w-1):
                if img[x,y] == 0:
                    dt[x,y] = min(dt[x,y], 4 + dt[x+1,y+1], 3 + dt[x,y+1], 4 + dt[x-1,y+1], 3 + dt[x+1,y])
            x = 0
            if img[x,y] == 0:
                dt[x,y] = min(dt[x,y], 4 + dt[x+1,y+1], 3 + dt[x,y+1], 3 + dt[x+1,y])
        return dt
    """
   
    def distance_transform_edt_np(self,og): #og is a numpy array of original image
        # -------------------------------------------------------------------------------------------
        # !!Convert to Pytorch!!
        # Source: https://stackoverflow.com/questions/53678520/speed-up-computation-for-distance-transform-on-image-in-python
        # ----------------------------------------------------------------------------------------------
        ones_loc = np.where(og == 1)
        #print(ones_loc[0].shape)
        ones = np.asarray(ones_loc).T # coords of all ones in og
        print("Numpy")
        print(ones.shape)
        zeros_loc = np.where(og == 0)
        zeros = np.asarray(zeros_loc).T # coords of all zeros in og
        print(zeros.shape)

        a = -2 * np.dot(zeros, ones.T) 
        print(a)
        b = np.sum(np.square(ones), axis=1) 
        c = np.sum(np.square(zeros), axis=1)[:,np.newaxis]
        dists = a + b + c
        dists = np.sqrt(dists.min(axis=1)) # min dist of each zero pixel to one pixel
        #print(f"dists={dists}")
        x = og.shape[0]
        y = og.shape[1]
        #print(f"x={x},y={y}")
        dist_transform = np.zeros((x,y))
        #print(len(dists))
        #print(len(zeros[:,0]))
        #print(len(zeros[:,1]))
        #print(len(zeros[:,0])+len(zeros[:,1]))
        dist_transform[zeros[:,0], zeros[:,1]] = dists 

        #print(dist_transform[np.where(dist_transform!=0)])

        #plt.figure()
        #plt.imshow(dist_transform)
        return dist_transform
  

    def distance_transform_edt(self, og):  # og is a PyTorch tensor of the original image
        """
        Source: https://stackoverflow.com/questions/53678520/speed-up-computation-for-distance-transform-on-image-in-python
        """
        ones_loc = torch.nonzero(og == 1, as_tuple=False).double()
        #print("Pytorch")
        #print(ones_loc.shape)
        zeros_loc = torch.nonzero(og == 0, as_tuple=False).double()

        a = -2 * torch.mm(zeros_loc, ones_loc.T)
        #print(torch.min(a))
        b = torch.sum(ones_loc**2, dim=1)
        #print(torch.min(b))
        c = torch.sum(zeros_loc**2, dim=1).unsqueeze(1)
        #print(torch.min(c))
        dists = a + b + c
        #print(a.shape)
        #print(dists.shape)
        #print(torch.min(dists))
        dists = torch.sqrt(dists.min(dim=1).values)  # min dist of each zero pixel to one pixel
        
        dist_transform = torch.zeros_like(og).double()# , dtype=torch.float)
        dist_transform[zeros_loc[:, 0].long(), zeros_loc[:, 1].long()] = dists

        return dist_transform




    def calculate_distances(self,segmentation_map,inverted_map):
        """
        Calculate the distance of each pixel to the closest and second closest nucleus.

        Parameters:
        segmentation_map (ndarray): Binary segmentation map where nuclei are marked with 1 and background with 0.

        Returns:
        tuple: Two ndarrays representing the distance to the closest nucleus (d1) and second closest nucleus (d2).
        """
        # Invert the segmentation map to get the background as 1 and nuclei as 0
        #inverted_map = np.logical_not(segmentation_map).astype(int)
        
        # Compute the distance transform
        #distance_transform = distance_transform_edt(inverted_map)
        #print("Distance Transform")
        #print(distance_transform)
        
        #print(inverted_map.detach().cpu().numpy())
        #distance_transform = self.distance_transform_edt(inverted_map)
        #print("Custom Distance Transform in Pytorch")
        #print(distance_transform)

        #distance_transform_distmap = euclidean_distance_transform(inverted_map)
        #print("Distmap Distance Transform in Pytorch")
        #print(distance_transform_distmap)

        #distance_transform_spicy = distance_transform_edt_spicy(inverted_map.detach().cpu().numpy())
        #print("Distance Transform in Spicy")
        #print(distance_transform_spicy)

        #c_distance_transform = self.chamfer_distance(inverted_map.detach().cpu().numpy())
        #print("Custom Chamfer Distance Transform in Pytorch")
        #print(c_distance_transform)

        #c_distance_transform_spicy = distance_transform_cdt_spicy(inverted_map.detach().cpu().numpy())
        #print("Chamfer Distance Transform in Spicy")
        #print(c_distance_transform_spicy)

        #print(f"Is equal = {distance_transform==distance_transform_spicy}")
        
        
        #distance_transform = self.distance_transform_edt_np(inverted_map.detach().cpu().numpy())
        # For the second closest distance, we need to mask out the closest nucleus and compute the distance again
        # Create a copy of the distance transform and set the closest nucleus distances to a large value
        #masked_distance_transform = np.copy(distance_transform)
        #masked_distance_transform[distance_transform == 0] = np.max(distance_transform)
        
        # Compute the distance transform again on the masked map
        #second_closest_distance_transform = self.distance_transform_edt_np(masked_distance_transform)
        
        #return distance_transform, second_closest_distance_transform

        #distance_transform = self.distance_transform_edt(inverted_map)
        """
        distmap.euclidean_distance_transform()
        Source: https://pypi.org/project/torch-distmap/ 
        !! It takes masks as an input and therefore does not allow backpropagation.
        """
        distance_transform = euclidean_distance_transform(inverted_map) #!! Does not support backpropagation 
        # Mask the distance transform
        masked_distance_transform = distance_transform.clone()
        #print(torch.isnan(distance_transform).any().item())
        #print(torch.max(distance_transform))
        masked_distance_transform[distance_transform == 0] = torch.max(distance_transform)
        #print(masked_distance_transform)

        # Compute the distance transform again on the masked map
        #second_closest_distance_transform = self.distance_transform_edt(masked_distance_transform)
        second_closest_distance_transform = euclidean_distance_transform(masked_distance_transform)
        return distance_transform, second_closest_distance_transform
    
    def wc(self,mask):
        ones = torch.nonzero(mask == 1, as_tuple=False).double().shape[0]
        zeros = torch.nonzero(mask == 0, as_tuple=False).double().shape[0]
        sc1 = 1/ones if ones>0 else 1
        sc0 = 1/zeros if zeros>0 else 1
        #scaled_mask = torch.where(mask==1,torch.mul(mask,sc1),torch.mul(mask,sc0))
        scaled_mask = torch.where(mask==1,sc1,-sc0)
        #print(ones_loc.shape[0]+zeros_loc.shape[0]==256**2)
        return scaled_mask

    def calculate_weight_map(self,H0, G, Gc, d1, d2, w0=10, sigma=4):
        """
        Calculate the stain-aware weight map w(x) using PyTorch.
        
        Parameters:
        H0 (Tensor): Hematoxylin channel normalized between 0 and 1.
        G (Tensor): Segmentation ground truth mask.
        Gc (Tensor): Complement of the segmentation ground truth mask.
        d1 (Tensor): Distance to the closest nucleus.
        d2 (Tensor): Distance to the second closest nucleus.
        w0 (float): Weight parameter.
        sigma (float): Sigma parameter for Gaussian function.
        
        Returns:
        Tensor: Stain-aware weight map w(x).
        """
        # Create a matrix with all entries being one
        J = torch.ones_like(H0)
        
        # Calculate H1 and H2
        H1 = (J - H0) * G
        H2 = H0 * Gc
        
        # Calculate wz(x)
        wz = self.wc(G) + (H1**3 + H2**(3.0/2.0))
        
        # Calculate the exponential component
        exp_component = w0 * torch.exp(- (d1 + d2)**2 / (2 * sigma**2)).to(d1.device)
        
        # Calculate the final weight map w(x)
        w = wz + exp_component
        
        return w
    
    def xentropy(self, input: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Assumes NCHW shape of array, must be torch.float32 dtype

        Args:
            <Shapes might be wrong here>
            input (torch.Tensor): Ground truth array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes
            target (torch.Tensor): Prediction array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes

        Returns:
            torch.Tensor: Cross entropy loss, with shape () [scalar], grad_fn = MeanBackward0
        """
        # reshape
        #input = input.permute(0, 2, 3, 1)
        #target = target.permute(0, 2, 3, 1)

        epsilon = 10e-8
        # scale preds so that the class probs of each sample sum to 1
        pred = input / torch.sum(input, -1, keepdim=True)
        # manual computation of crossentropy
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
        #print(w.shape)
        #print(target.shape)
        #print(input.shape)
        loss = -torch.sum((w * torch.log(pred)), -1, keepdim=True)
        loss = loss.mean() if self.reduction == "mean" else loss.sum()
        #print(loss.shape)

        return loss

    def forward(
        self,
        input: List[torch.Tensor],
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        channel 0 - background
        channel 1 - nuclei
        """
        #inp = target[0][0]
        #print(inp.shape)
        #inp = torch.mul(inp,255)
        #inp = inp.type(torch.int)
        #img = inp.detach().cpu().numpy()
        #save_image(inp,'target_img_0.png')

        #mask = self.classify_nucleus(inp)
        #print(mask.shape)
        img_batch = input[1]
        preds = input[0]
        sams_loss = 0
        #print(f"img_batch: {img_batch.requires_grad}")
        #print(f"preds: {preds.requires_grad}")
        #print(f"Target: {target.requires_grad}")
        for i,map in enumerate(target):
            G = map[1]
            G_mask = self.classify_nucleus(map)
            #print("Nuclei Binary Map/Segmentation GT Mask")
            #print(G)
            # Invert the segmentation map to get the background as 1 and nuclei as 0
            ones = torch.ones_like(G).to(G.device)
            #print("Ones")
            #print(ones)
            Gc = torch.sub(ones,G)
            Gc_mask = torch.sub(ones,G_mask)
            #print("Complement of Nuclei Binary Map/Segmentation GT Mask")
            #print(Gc)
            #print(G[0]+Gc[0])
            #break
            #print(Gc)
            #print(f"G: {G.requires_grad}")
            d1, d2 = self.calculate_distances(G_mask,Gc_mask)
            #print(f"d1={d1},d2={d2}")
            #print(torch.max(d1))
            #print(torch.max(d2))
            H0 = self.get_H_channel(img_batch[i])
            if np.max(H0)>np.min(H0):
                H0 = torch.from_numpy((H0-np.min(H0))/(np.max(H0)-np.min(H0))).to(map.device) # normalize between 0 and 1
            else:
                H0 = torch.from_numpy(H0).to(map.device)
            #print(H0.shape)
            w = self.calculate_weight_map(H0=H0,G=G,Gc=Gc,d1=d1,d2=d2)
            #inp = torch.unsqueeze(preds[i],0)
            #tar = torch.unsqueeze(target[i],0)
            inp = preds[i][1]
            tar = G
            #inp = torch.unsqueeze(img_batch[i][1],0)
            #tar = torch.unsqueeze(G,0)
            #w = torch.unsqueeze(w,0)
            loss = self.xentropy(input=inp,target=tar,w=w)
            loss = torch.squeeze(loss)
            #print(inp.device)
            #print("SAMS Loss")
            #print(f"inp: {inp.requires_grad}")
            #print(f"w: {w.requires_grad}")
            #print(f"loss: {loss.requires_grad}")
            #sams_loss += torch.sum(torch.mul(w,loss))
            sams_loss += loss
            #break
        #print(f"sams_loss: {sams_loss.requires_grad}")
        return sams_loss
        
# combined with cross entropy loss, instance level
class LossVariance(nn.Module):
    """ 
    Doesn't support backpropagation!!!
    The instances in target should be labeled 
    """
    def __init__(self):
        super(LossVariance, self).__init__()
    
    def _get_labels(self,arr):
        """
        For each sample in the batch, select the maximum value among the 6 channels for each pixel.

        Parameters:
        arr (torch.Tensor): A tensor of shape (6, 256, 256) where B is the batch size.

        Returns:
        torch.Tensor: A tensor of shape (256, 256) where each pixel contains the maximum value 
                    across the 6 channels.
        """
        # Apply torch.max to get the maximum value along the channel dimension (dim=0)
        #print(arr.shape)
        max_values, max_indices = torch.max(arr, dim=0)
        #print(max_values.device)
        
        return max_indices

    def _get_probs(self,arr):
        """
        For each sample in the batch, select the maximum value among the 6 channels for each pixel.

        Parameters:
        arr (torch.Tensor): A tensor of shape (6, 256, 256) where B is the batch size.

        Returns:
        torch.Tensor: A tensor of shape (256, 256) where each pixel contains the maximum value 
                    across the 6 channels.
        """
        # Apply torch.max to get the maximum value along the channel dimension (dim=0)
        #print(arr.shape)
        max_values, max_indices = torch.max(arr, dim=0)
        #print(max_values.device)
        
        return max_values

    def forward(self, input, target):
        #print(input.shape)
        B = input.size(0)

        loss = 0
        for k in range(B):
            #print(target[k].shape)
            #break
            t = self._get_labels(target[k]) # Get labels from 6 dimensional vector
            #i = self._get_labels(input[k])
            unique_vals = t.unique()
            #unique_vals_inp = i.unique()
            #print(unique_vals)
            #break
            unique_vals = unique_vals[unique_vals != 0]
            # print(unique_vals.shape)

            sum_var = 0
            for val in unique_vals:
                instance = input[k][:,t == val] 
                instance = instance.float()
                #print(instance.shape)
                if instance.size(0) > 1:
                    sum_var += instance.var(dim=0).sum()
            #print(i.requires_grad)
            #break
            loss += sum_var / (len(unique_vals) + 1e-8)
        loss /= B
        #print(loss.requires_grad)
        #print(loss.device)
        return loss

class HuberLossMaps(_Loss):
    """Calculate Psudo-Huber (smooth approximation) loss for combined horizontal and vertical maps of segmentation tasks."""

    def __init__(self,delta) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")
        self.delta = 0.5

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss calculation

        Args:
            input (torch.Tensor): Prediction of combined horizontal and vertical maps
                with shape (N, 2, H, W), channel 0 is vertical and channel 1 is horizontal
            target (torch.Tensor): Ground truth of combined horizontal and vertical maps
                with shape (N, 2, H, W), channel 0 is vertical and channel 1 is horizontal

        Returns:
            torch.Tensor: Mean squared error per pixel with shape (N, 2, H, W), grad_fn=SubBackward0

        """
        # reshape
        a = input - target
        loss = (a*a).mean() # Change
        return loss

def retrieve_loss_fn(loss_name: dict, **kwargs) -> _Loss:
    """Return the loss function with given name defined in the LOSS_DICT and initialize with kwargs

    kwargs must match with the parameters defined in the initialization method of the selected loss object

    Args:
        loss_name (dict): Name of the loss function

    Returns:
        _Loss: Loss
    """
    loss_fn = LOSS_DICT[loss_name]
    loss_fn = loss_fn(**kwargs)

    return loss_fn


LOSS_DICT = {
    "xentropy_loss": XentropyLoss,
    "dice_loss": DiceLoss,
    "mse_loss_maps": MSELossMaps,
    "msge_loss_maps": MSGELossMaps,
    "FocalTverskyLoss": FocalTverskyLoss,
    "MCFocalTverskyLoss": MCFocalTverskyLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,  # input logits, targets
    "L1Loss": nn.L1Loss,
    "MSELoss": nn.MSELoss,
    "CTCLoss": nn.CTCLoss,  # probability
    "NLLLoss": nn.NLLLoss,  # log-probabilities of each class
    "PoissonNLLLoss": nn.PoissonNLLLoss,
    "GaussianNLLLoss": nn.GaussianNLLLoss,
    "KLDivLoss": nn.KLDivLoss,  # argument input in log-space
    "BCELoss": nn.BCELoss,  # probabilities
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,  # logits
    "MarginRankingLoss": nn.MarginRankingLoss,
    "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
    "MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
    "HuberLoss": nn.HuberLoss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "SoftMarginLoss": nn.SoftMarginLoss,  # logits
    "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
    "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
    "MultiMarginLoss": nn.MultiMarginLoss,
    "TripletMarginLoss": nn.TripletMarginLoss,
    "TripletMarginWithDistanceLoss": nn.TripletMarginWithDistanceLoss,
    "MAEWeighted": MAEWeighted,
    "MSEWeighted": MSEWeighted,
    "BCEWeighted": BCEWeighted,  # logits
    "CEWeighted": CEWeighted,  # logits
    "L1LossWeighted": L1LossWeighted,
    "BendingLoss": BendingLoss, # new
    "SAMSLoss": SAMSLoss, # new
    "LossVariance": LossVariance, # new
    "huber_loss_maps": nn.HuberLoss, # new
}