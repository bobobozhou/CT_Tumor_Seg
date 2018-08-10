import numpy as np
import os
from skimage.filters import threshold_otsu
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage import feature
import torch
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy import ndimage
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import cv2
import random
import SimpleITK as sitk
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


'''Metrics for evaluation of segmentation performance'''
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
        # vol_output = correct_pred_vol(vol_output, ratio=0.1)

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


def metric_VS_volume(output, target, ind_all):
    """ Calculation of DSC with respect to volume using the slice index (correspond to case) """
    num_volume = ind_all.max()
    all_VS_volume = []

    for i in range(ind_all.min(), ind_all.max() + 1):
        vol_gt = target[np.where(ind_all == i)[0], :, :]
        
        vol_output_prob = output[np.where(ind_all == i)[0], :, :]
        vol_output = prob_to_segment(vol_output_prob)
        # vol_output = correct_pred_vol(vol_output, ratio=0.1)

        vs = VS(vol_output, vol_gt, empty_score=1.0)
        all_VS_volume.append(vs)

    all_VS_volume = np.array(all_VS_volume)
    mVS = all_VS_volume.mean()
    return [mVS], [all_VS_volume]


def VS(im_pred, im_target, empty_score=1.0):
    """
    Computes the Volumetric Similarity, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.

    Returns
    -------
    vs : float
        volumetric similiarity as a float on range [0,1].
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

    # Compute Volumtric Similarity
    intersection = np.logical_and(im_pred, im_target)
    FN = np.logical_xor(im_pred, intersection)
    FP = np.logical_xor(im_target, intersection)

    return 1. - np.abs(1.*FN.sum() - 1.*FP.sum()) / im_sum


def metric_HD_volume(output, target, ind_all):
    """ Calculation of DSC with respect to volume using the slice index (correspond to case) """
    num_volume = ind_all.max()
    all_HD_volume = []

    for i in range(ind_all.min(), ind_all.max() + 1):
        vol_gt = target[np.where(ind_all == i)[0], :, :]
        
        vol_output_prob = output[np.where(ind_all == i)[0], :, :]
        vol_output = prob_to_segment(vol_output_prob)
        # vol_output = correct_pred_vol(vol_output, ratio=0.1)

        hd = HD(vol_output, vol_gt)
        all_HD_volume.append(hd)

    all_HD_volume = np.array(all_HD_volume)
    mHD = all_HD_volume.mean()
    return [mHD], [all_HD_volume]


def HD(im_pred, im_target, empty_score=0):
    """
    Computes the Hausdorff Distance, a measure of set.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.

    Returns
    -------
    hd : float
        measure the subtle segmentation performance
    """

    # threshold the predicted segmentation image (a probablity map)
    im_pred = prob_to_segment(im_pred)

    # the targert segmentation image
    im_target = np.asarray(im_target).astype(np.bool)

    # calculate the dice using the 1. prediction and 2. ground truth
    if im_pred.shape != im_target.shape:
        raise ValueError("Shape mismatch: im_pred and im_target must have the same shape!")

    im_sum = im_pred.sum() + im_target.sum()
    if im_sum == 0 or im_pred.sum() == 0 or im_target.sum() == 0:
        return empty_score

    # Compute Hausdorff Distance
    hausdorff_distance_image_filter = sitk.HausdorffDistanceImageFilter()
    V1 = sitk.GetImageFromArray(im_pred.astype(np.int16))
    V2 = sitk.GetImageFromArray(im_target.astype(np.int16))
    hausdorff_distance_image_filter.Execute(V1, V2)
    hd = hausdorff_distance_image_filter.GetHausdorffDistance()

    return hd


'''Make segmentation visualization on the tensorboard'''
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


""" Fully-connected Conditional Random Field for refine the segmentation results """
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


""" Save segmentation results for paper"""
def save_seg_montage(input, output, target, ind_all, savefolder):
    """
    make the Montage for saving the slice + segmentation contour results
    Parameters
    ----------
    output: array-like
        Any array of arbitrary size from the model prediction with case-index.
    targte:  
        Any array of arbitrary size (ground truth segmentation) with case-index
        * Need to be same size as output
    """

    if output.shape != target.shape:
        raise ValueError("Shape mismatch: Image & Prediction & Ground-Truth must have the same shape!")

    for i in range(ind_all.min(), ind_all.max() + 1):
        vol_input = input[np.where(ind_all == i)[0], :, :]
        vol_gt = target[np.where(ind_all == i)[0], :, :]
        vol_output_prob = output[np.where(ind_all == i)[0], :, :]
        vol_output = prob_to_segment(vol_output_prob)

        vol_input = vol_input.transpose(1, 2, 0)
        vol_gt = vol_gt.transpose(1, 2, 0)
        vol_output = vol_output.transpose(1, 2, 0)

        plot_data_3d(vol_input, vol_output, vol_gt, savepath=savefolder + str(i) + '.png')

    return 


def plot_data_3d(vol_input, vol_output, vol_gt, savepath):
    """
    Generate an image for 3D data.
    1) show the corresponding 2D slices.
    """

    # Draw
    slides = plot_slides(vol_input, vol_output, vol_gt)

    # Save
    cv2.imwrite(savepath, slides)


def plot_slides(vol_input, vol_output, vol_gt, _range=None, colored=False):
    """Plot the 2D slides of 3D data"""
    v = vol_input
    vol_output = vol_output.astype(np.uint8)
    vol_gt = vol_gt.astype(np.uint8)

    # Rescale the value of voxels into [0, 255], as unsigned byte
    if _range == None:
        v_n = v / (np.max(np.abs(v)) + 0.0000001)
        v_n = (128 + v_n * 127).astype(np.uint8)

    else:
        v_n = (v - _range[0]) / (_range[1] - _range[0])
        v_n = (v_n * 255).astype(np.uint8)

    # Plot the slides
    h, w, d = v.shape
    side_w = int(np.ceil(np.sqrt(d)))
    side_h = int(np.ceil(float(d) / side_w))

    board = np.zeros(((h + 1) * side_h, (w + 1) * side_w, 3))
    for i in range(side_h):
        for j in range(side_w):
            if i * side_w + j >= d:
                break

            img = v_n[:, :, i * side_w + j]
            img = np.repeat(img[:,:,np.newaxis], 3, axis=2)

            contours_pred, _ = cv2.findContours(vol_output[:, :, i * side_w + j].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_gt, _ = cv2.findContours(vol_gt[:, :, i * side_w + j].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours_pred, -1, (0, 200, 0), 1)
            cv2.drawContours(img, contours_gt, -1, (0, 0, 200), 1)

            board[(h + 1) * i + 1: (h + 1) * (i + 1), (w + 1) * j + 1: (w + 1) * (j + 1), :] = img

    # Return a 2D array representing the image pixels
    return board.astype(int)


'''Generate the training list .txt : initial with only mid-slice = dis_index equal to 0'''
def create_semitrainlist(train_list_filename, semitrain_list_dir, dis_ind_set, n):
    semitrain_list_filename = semitrain_list_dir + 'semitrain_list' + '{:02d}'.format(n) + '.txt'
    f_semi_list = open(semitrain_list_filename, 'wt') 
    f_train_list = open(train_list_filename)

    for line in f_train_list.readlines():
        items = line.split()

        if int(float(items[1])) in dis_ind_set:
            f_semi_list.write(line)

    f_train_list.close()
    f_semi_list.close()

    return semitrain_list_filename


'''Predict & Save the unlabel data results for semi-training'''
def predict_unlabel(unlabel_loader, model, percent_keep, network='UNet'):
    # switch to evaluation mode and evaluate
    model.eval()
    for i, (case_index, dis_index, input_name, input, input_raw) in enumerate(unlabel_loader):
        input_var = torch.autograd.Variable(input, requires_grad=False).cuda()
        input_raw_var = torch.autograd.Variable(input_raw, requires_grad=False).cuda()

        # 1) output BOUNDARY, REGION, FINAL_REGION from models
        if network is 'SiBA':
             _, _, _, _, _, _, _, _, output_fin = model(input_var)
        if network is 'UNet':
            output_fin = model(input_var)
        if network is 'HNN':
            output_fin = model(input_var)
        if network is 'WSSS':
            output_fin = model(input_var)

        # 2) store: case_ind, dis_ind, input_img_name, input_img, output_mask
        if i == 0:
            case_index_all = case_index
            dis_index_all = dis_index
            input_name_all = input_name
            input_raw_all = input_raw_var.data.cpu().numpy()[:,0,:,:]
            output_mask_all = generate_CRF(img=input_var.data.cpu().numpy()[:,0,:,:], 
                                           pred=output_fin.data.cpu().numpy()[:,0,:,:],
                                           iter=10, n_labels=2)

        else:
            case_index_all = np.concatenate((case_index_all, case_index), axis=0)
            dis_index_all = np.concatenate((dis_index_all, dis_index), axis=0)
            input_name_all = np.concatenate((input_name_all, input_name), axis=0)
            input_raw_all = np.concatenate((input_raw_all, input_raw_var.data.cpu().numpy()[:,0,:,:]), axis=0)
            output_mask_all = np.concatenate((output_mask_all, generate_CRF(img=input_var.data.cpu().numpy()[:,0,:,:],
                                                                            pred=output_fin.data.cpu().numpy()[:,0,:,:],
                                                                            iter=10, n_labels=2)), axis=0)
    # generate edge from predicted mask
    output_edge_all = np.zeros((output_mask_all.shape))
    for ii in range(output_mask_all.shape[0]):
        output_mask = output_mask_all[ii].astype(np.float32)
        output_edge = feature.canny(output_mask).astype(np.uint16)
        output_edge_all[ii, :, :] = output_edge

    # select certain percentage from the data
    l = output_mask_all.shape[0]
    l_new = int(l * percent_keep)
    ind_new = np.unique(random.sample(xrange(l), l_new))

    case_index_all = case_index_all[ind_new]
    dis_index_all = dis_index_all[ind_new]
    input_name_all = input_name_all[ind_new]
    input_raw_all = input_raw_all[ind_new]
    output_mask_all = output_mask_all[ind_new]
    output_edge_all = output_edge_all[ind_new]

    return case_index_all, dis_index_all, input_name_all, input_raw_all, output_mask_all, output_edge_all


def save_unlabel(case_ind_unlabel, dis_ind_unlabel, 
                 img_save_dir, img_name_unlabel,
                 mask_save_dir, mask_unlabel, 
                 edge_save_dir, edge_unlabel,
                 last_semitrain_list_filename,
                 n):
    """
    create/add the unlabelled data (image, mask, edge) to corresponding txt file

    """

    if case_ind_unlabel.shape[0] != dis_ind_unlabel.shape[0] != img_name_unlabel.shape[0] != mask_unlabel.shape[0] != edge_unlabel.shape[0]:
        raise ValueError("Number of case mismatch: image, mask, image name must have the same shape!")

    # add last trianing list to current training list
    new_semitrain_list_filename = last_semitrain_list_filename.replace('{:02d}'.format(n), '{:02d}'.format(n+1)) 

    f_new = open(new_semitrain_list_filename, 'wt') 
    f_last = open(last_semitrain_list_filename)

    for line in f_last.readlines():
        f_new.write(line)

    # add new training list to current training list & Save the new data into folder
    for i in range(img_name_unlabel.shape[0]):
        case_ind = case_ind_unlabel[i]
        dis_ind = dis_ind_unlabel[i]
        img_name = img_name_unlabel[i].replace(img_save_dir + '/', '')
        mask_name = img_name.replace('img', 'mask')
        edge_name = img_name.replace('img', 'edge')

        mask = mask_unlabel[i].astype(np.uint16)
        edge = edge_unlabel[i].astype(np.uint16)

        cv2.imwrite(os.path.join(mask_save_dir, mask_name), mask)
        cv2.imwrite(os.path.join(edge_save_dir, edge_name), edge)

        f_new.write(str(case_ind) + ' ' + '{:03.2f}'.format(dis_ind) + ' ' + img_name + ' ' + mask_name + ' ' + edge_name + ' 0 0 0 0 ' + '\r\n')

    f_new.close()
    f_last.close()
    
    return new_semitrain_list_filename