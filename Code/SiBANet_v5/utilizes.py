import numpy as np
from skimage.filters import threshold_otsu
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy import ndimage
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import cv2
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
        # vol_output = correct_pred_vol(vol_output, ratio=0.1)

        montage_input = vol_to_montage(vol_input)
        montage_gt = vol_to_montage(vol_gt)
        montage_output = vol_to_montage(vol_output)

        montage_input = np.repeat(montage_input[np.newaxis, np.newaxis, :, :], 3, axis=1)
        montage_gt = np.repeat(montage_gt[np.newaxis, np.newaxis, :, :], 3, axis=1)
        montage_output = np.repeat(montage_output[np.newaxis, np.newaxis, :, :], 3, axis=1)

        disp_mat = np.concatenate((montage_input, montage_output, montage_gt), axis=0)
        dict[i] = disp_mat

    return dict

"""threshold slice or volume to binary as segmentation"""
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

""" Make Montage display for volumetric data """
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

""" Correction of volume slices progressively starting from the middle slice """
def eliminate_region(image, ratio=0.1):
    # detect regions and labels
    label_img = label(image)
    regions = regionprops(label_img)

    # run eliminate process only if there is more than one region
    if len(regions) > 1:
        areas = []
        labs = []
        for i, props in enumerate(regions):
            area = props.area
            areas.append(area)

            lab = np.zeros(props._label_image.shape)
            lab[props._label_image == i+1] = 1
            labs.append(lab)

        ratios = np.array(areas, dtype=np.float) / max(areas)
        index = np.where(ratios > ratio)[0]
        labs_new = [labs[i]  for i in index]
        labs_new = np.array(labs_new)

        image_new = np.sum(labs_new, axis=0)

    else:
        image_new = image

    return image_new


def merge_region(image):
    # detect regions and labels
    label_img = label(image)
    regions = regionprops(label_img)

    # create connection mask, if there is more than 2 regions
    img_conn = Image.new('L', (image.shape[0], image.shape[1]), 0)

    if len(regions) > 1:
        polygon_pts = []
        area_min = float("inf")
        for props in regions:
            y0, x0 = props.centroid
            polygon_pts.append((x0, y0))

            # store the structure element for dilation
            if props.area < area_min:
                area_min = props.area
                se = props.convex_image

        ImageDraw.Draw(img_conn).polygon(polygon_pts, outline=1, fill=1)
        img_conn = np.array(img_conn)
        img_conn = binary_dilation(img_conn, structure=se)

        image_new = np.array(np.logical_or(image, img_conn), dtype=np.float)

    else:
        image_new = image

    return image_new


def correct_image_slice(img_pre, img_cur, Abot=0.2, Atop=1.3, Ctop=1):
    # detect regions and labels
    label_img_pre = label(img_pre)
    regions_pre = regionprops(label_img_pre)

    label_img_cur = label(img_cur)
    regions_cur = regionprops(label_img_cur)

    # check if there is more than 1 region
    if len(regions_pre) > 1 or len(regions_cur) > 1:
        raise ValueError("There should be only 1 binary region or no binary region in the image")

    if len(regions_pre) == 0:
        raise ValueError("previous image shouldn't be empty")

    # process and generate current slice based on last slice
    if len(regions_cur) == 0:    # if empty, directly copy over
        if np.count_nonzero(binary_erosion(img_pre, iterations=1)) <= 20:
            img_cur_new = img_pre
        else:
            img_cur_new = binary_erosion(img_pre, iterations=1)

    else:   # if not empty, determine whether to keep or replace           
        regions_pre = regions_pre[0]
        regions_cur = regions_cur[0]

        # check 1) area ratio ; 2) centroid location
        A_ratio = float(regions_cur.area) / float(regions_pre.area)

        x_cur, y_cur = regions_cur.centroid
        x_pre, y_pre = regions_pre.centroid
        C_dis = ((x_cur - x_pre)**2 + (y_cur - y_pre)**2)**(1.0/2.0)  # center distance
        S_pre = (regions_pre.major_axis_length + regions_pre.minor_axis_length) / 2.0 / 2.0  # radius of the pre region 
        C_ratio = C_dis / S_pre

        if Abot <= A_ratio <= Atop and C_ratio <= Ctop:
            img_cur_new = img_cur
        else:
            if np.count_nonzero(binary_erosion(img_pre, iterations=1)) <= 20:
                img_cur_new = img_pre
            else:
                img_cur_new = binary_erosion(img_pre, iterations=1)

    img_cur_new = eliminate_region(img_cur_new, ratio=0.1)
    img_cur_new = merge_region(img_cur_new)
    return img_cur_new


def correct_pred_vol(vol, ratio=0.1):
    '''
    vol: the predicted segmentation volume from the model
    ratio: ratio threshold for eliminating small regions
    '''

    vol_new = vol.copy()

    try:
        # process middle slice
        ind_mid = int(vol.shape[0] / 2)
        img_mid = vol[ind_mid, :, :]
        img_mid = eliminate_region(img_mid, ratio=0.1)
        img_mid = merge_region(img_mid)
        vol_new[ind_mid, :, :] = img_mid

        # go upper and process
        upper_list = range(ind_mid + 1, vol.shape[0])
        for i in upper_list:
            img_cur = vol[i, :, :]
            img_cur = eliminate_region(img_cur, ratio=0.1)
            img_cur = merge_region(img_cur)

            img_pre = vol_new[i - 1, :, :]

            img_cur_new = correct_image_slice(img_pre, img_cur)
            vol_new[i, :, :] = img_cur_new

        # go bottom and process
        bottom_list = list(reversed(range(0, ind_mid)))
        for i in bottom_list:
            img_cur = vol[i, :, :]
            img_cur = eliminate_region(img_cur, ratio=0.1)
            img_cur = merge_region(img_cur)

            img_pre = vol_new[i + 1, :, :]
            img_cur_new = correct_image_slice(img_pre, img_cur)
            vol_new[i, :, :] = img_cur_new

        return vol_new

    except:
        return vol_new


""" Save segmentation results for paper"""
def save_seg_montage(input, output, target, ind_all):
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

        plot_data_3d(vol_input, vol_output, vol_gt, savepath='./_RESULTS/' + str(i) + '.png')

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