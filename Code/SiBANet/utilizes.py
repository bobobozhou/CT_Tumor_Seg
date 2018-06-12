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


def metric_DSC_volume(output, target, ind_all):
    """ Calculation of DSC with respect to volume using the slice index (correspond to case) """
    num_volume = ind_all.max()
    all_DSC_volume = []

    for i in range(ind_all.min(), ind_all.max() + 1):
        vol_gt = target[np.where(ind_all == i)[0], :, :]
        
        vol_output_prob = output[np.where(ind_all == i)[0], :, :]
        vol_output = prob_to_segment(vol_output_prob)

        dice = DICE(vol_output, vol_gt, empty_score=1.0)
        all_DSC_volume.append(dice)

    all_DSC_volume = np.array(all_DSC_volume)
    mDSC = all_DSC_volume.mean()
    return [mDSC], [all_DSC_volume]


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
    im_pred = prob_to_segment(im_pred)

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


def make_tf_disp_slice(output, target):
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


def make_tf_disp_volume(input, output, target, ind_all):
    """
    make the Montage for tensorboard to display 3D volume
    Parameters
    ----------
    output: array-like
        Any array of arbitrary size from the model prediction with case-index.
    targte:  
        Any array of arbitrary size (ground truth segmentation) with case-index
        * Need to be same size as output

    Return
    ----------
    dictionary for inputting the tf logger function, displaying montage
    """

    if output.shape != target.shape:
        raise ValueError("Shape mismatch: Image & Prediction & Ground-Truth must have the same shape!")

    dict = {}
    for i in range(ind_all.min(), ind_all.max() + 1):
        vol_input = input[np.where(ind_all == i)[0], :, :]
        vol_gt = target[np.where(ind_all == i)[0], :, :]
        vol_output_prob = output[np.where(ind_all == i)[0], :, :]
        vol_output = prob_to_segment(vol_output_prob)

        montage_input = vol_to_montage(vol_input)
        montage_gt = vol_to_montage(vol_gt)
        montage_output = vol_to_montage(vol_output)

        montage_input = np.repeat(montage_input[np.newaxis, np.newaxis, :, :], 3, axis=1)
        montage_gt = np.repeat(montage_gt[np.newaxis, np.newaxis, :, :], 3, axis=1)
        montage_output = np.repeat(montage_output[np.newaxis, np.newaxis, :, :], 3, axis=1)

        disp_mat = np.concatenate((montage_input, montage_output, montage_gt), axis=0)
        dict[i] = disp_mat

    return dict


def prob_to_segment(prob):
    """
    threshold the predicted segmentation image (a probablity image/volume)
    using 0.5 hard thresholding
    """
    thresh = threshold_otsu(prob)
    # thresh = 0.5
    seg = prob > thresh
    seg = np.asarray(seg).astype(np.bool)

    return seg


def vol_to_montage(vol):
    n_slice, w_slice, h_slice = np.shape(vol)
    nn = int(np.ceil(np.sqrt(n_slice)))
    mm = nn
    M = np.zeros((mm * h_slice, nn * w_slice)) 

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= n_slice: 
                break
            sliceM = j * w_slice 
            sliceN = k * h_slice
            M[sliceN:sliceN + w_slice, sliceM:sliceM + h_slice] = vol[image_id, :, :]
            image_id += 1

    return M
