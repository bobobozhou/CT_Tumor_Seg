import os
import numpy as np
import ipdb
# import matplotlib.pyplot as plt

from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.misc import imsave

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

def correct_image_slice(img_pre, img_cur, Abot=0.7, Atop=1.2, Ctop=1):
    # detect regions and labels
    label_img_pre = label(img_pre)
    regions_pre = regionprops(label_img_pre)

    label_img_cur = label(img_cur)
    regions_cur = regionprops(label_img_cur)

    # check if there is more than 1 region
    if len(regions_pre) > 1 or len(regions_cur) > 1:
        raise ValueError("There should be only 1 binary region or no binary region in the image")

    # process and generate current slice based on last slice
    if len(regions_cur) == 0:     # if empty, directly copy over
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
            img_cur_new = binary_erosion(img_pre, iterations=1)

    return img_cur_new

if __name__ == '__main__':
    # create simulated images
    vol = np.array(np.load('8.npy'), dtype=np.float)
    vol_new = vol.copy()

    # process middle slice
    ind_mid = int(vol.shape[0] / 2)
    img_mid = vol[ind_mid, :, :]
    img_mid = eliminate_region(img_mid, ratio=0.1)
    img_mid = merge_region(img_mid)
    vol_new[ind_mid, :, :] = img_mid

    # go upper and process
    upper_list = range(ind_mid + 1, vol.shape[0])
    for i in upper_list:
        print i
        img_cur = vol[i, :, :]
        img_cur = eliminate_region(img_cur, ratio=0.1)
        img_cur = merge_region(img_cur)

        img_pre = vol_new[i - 1, :, :]

        img_cur_new = correct_image_slice(img_pre, img_cur)
        vol_new[i, :, :] = img_cur_new

    # go bottom and process
    bottom_list = list(reversed(range(0, ind_mid)))
    for i in bottom_list:
        print i
        img_cur = vol[i, :, :]
        img_cur = eliminate_region(img_cur, ratio=0.1)
        img_cur = merge_region(img_cur)

        img_pre = vol_new[i + 1, :, :]
        img_cur_new = correct_image_slice(img_pre, img_cur)
        vol_new[i, :, :] = img_cur_new

    # save
    imsave(os.path.join('saved','img_org.jpg'), vol[2,:,:])
    imsave(os.path.join('saved','img_new.jpg'), vol_new[2,:,:])