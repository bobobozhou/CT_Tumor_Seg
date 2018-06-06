import numpy as np
from skimage.filters import threshold_otsu
import ipdb

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def metric_DSC_slice(output, target):
    """ Calculation of DSC with respect to slice """
    num_slice = target.shape[0]
    all_DSC_slice = []

    for i in range(num_slice):
        gt = target[i, 0, :, :].astype('float32')
        pred = output[i, 0, :, :].astype('float32')

        dice = DICE(gt, pred, empty_score=1.0)
        all_DSC_slice.append(dice)

    all_DSC_slice = np.array(all_DSC_slice)
    mDSC = all_DSC_slice.mean()
    return [mDSC], [all_DSC_slice]


def metric_DSC_volume(output, target):
    """ Calculation of DSC with respect to  """
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()

    num_class = target.shape[1]
    all_roc_auc = []
    for cid in range(num_class):
        gt_cls = target_np[:, cid].astype('float32')
        pred_cls = output_np[:, cid].astype('float32')

        if all(v == 0 for v in gt_cls):
            roc_auc = float('nan')
        else:
            roc_auc = roc_auc_score(gt_cls, pred_cls, average='weighted')

        all_roc_auc.append(roc_auc)

    all_roc_auc = np.array(all_roc_auc)
    mROC_AUC = all_roc_auc[~np.isnan(all_roc_auc)].mean()
    return [mROC_AUC], [all_roc_auc]


def DICE(im_pred, im_target, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.

    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
    """

    # threshold the predicted segmentation image (a probablity map)
    thresh = threshold_otsu(im_pred)
    im_pred = im_pred > thresh
    im_pred = np.asarray(im_pred).astype(np.bool)

    # the targert segmentation image
    im_target = np.asarray(im_target).astype(np.bool)

    # calculate the dice using the 1. prediction and 2. ground truth
    if im_pred.shape != im_target.shape:
        raise ValueError("Shape mismatch: im_pred and im_target must have the same shape!")

    im_sum = im_pred.sum() + im_target.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im_pred, im_target)

    return 2. * intersection.sum() / im_sum


def make_tf_disp(output, target):
    """
    make the numpy matrix for tensorboard to display
    Parameters
    ----------
    output: array-like
        Any array of arbitrary size from the model prediction.
    targte:  
        Any array of arbitrary size (ground truth segmentation)
        * Need to be same size as output

    Return
    ----------
    4D array for inputting the tf logger function
    """

    if output.shape != target.shape:
        raise ValueError("Shape mismatch: Prectiction and Ground-Truth must have the same shape!")

    output = np.repeat(output[np.newaxis, np.newaxis, :, :], 3, axis=1)
    target = np.repeat(target[np.newaxis, np.newaxis, :, :], 3, axis=1)
    disp_mat = np.concatenate((output, target), axis=0)

    return disp_mat

