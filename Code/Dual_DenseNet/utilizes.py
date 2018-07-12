import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from sklearn.metrics import roc_auc_score, average_precision_score
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


def metric_ROC(output, target):
    """ Calculation of min Ap """
    output_np = output
    target_np = target

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


def metric_DSC_slice(output, target):
    """ Calculation of DSC with respect to slice """
    num_slice = target.shape[0]
    all_DSC_slice = []

    for i in range(num_slice):
        gt = target[i, :, :].astype('float32')
        pred = output[i, :, :].astype('float32')

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
    if len(np.unique(prob)) is 1:
        thresh = 0.5
    else:
        thresh = threshold_otsu(prob)

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


def generate_CRF(img, pred, iter=20, n_labels=2):
    '''
    INPUT
    ----------------------------------------
    img: nimages x h x w
    pred: nimages x h x 2
    iter: number of iteration for CRF inference
    n_labels: number of class

    RETUREM
    ----------------------------------------
    map: label generated from CRF using Unary & Image
    '''

    for i in range(img.shape[0]):
        # prepare the image and prediction
        img_ind = img[i, :, :][:, :, np.newaxis]
        pred_ind = np.tile(pred[i, :, :][np.newaxis, :, :], (2, 1, 1))
        pred_ind[1, :, :] = 1 - pred_ind[0, :, :]
        # ipdb.set_trace()

        # setup the dense conditional random field for segmentation
        d = dcrf.DenseCRF2D(img_ind.shape[1], img_ind.shape[0], n_labels)

        U = unary_from_softmax(pred_ind)  # note: num classes is first dim
        d.setUnaryEnergy(U)

        pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=img_ind, chdim=2)
        d.addPairwiseEnergy(pairwise_energy, compat=500)
        d.addPairwiseGaussian(sxy=(3, 3), compat=500, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        # run iterative inference to do segmentation
        Q, tmp1, tmp2 = d.startInference()
        for _ in range(iter):
            d.stepInference(Q, tmp1, tmp2)
        map_crf = 1 - np.argmax(Q, axis=0).reshape((img_ind.shape[1], img_ind.shape[0]))
        # kl = d.klDivergence(Q) / (img_ind.shape[1] * img_ind.shape[0])

        if np.count_nonzero(map_crf) <= 5:
            map_crf = pred[i, :, :]

        # post-proces the binary segmentation (dilate)
        map_crf = ndimage.binary_dilation(map_crf, iterations=2)

        # save in the array and output
        if i == 0:
            map_all = map_crf[np.newaxis, :, :]
        else:
            map_all = np.concatenate((map_all, map_crf[np.newaxis, :, :]), axis=0)

    return map_all