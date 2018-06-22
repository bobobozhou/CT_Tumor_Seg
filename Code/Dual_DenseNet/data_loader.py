import torch
import numpy as np
from torch.utils.data import Dataset
import transforms_3pair.transforms_3pair as transforms_3pair    # customizd transform for applying same random for 3 images (image, mask, edge)
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
import os
import ipdb


class CTTumorDataset_R(Dataset):
    """"CT Tumor 2D Data loader"""

    def __init__(self, image_data_dir, mask_data_dir, list_file, transform=None, norm_img=None, norm_cond=None):

        """
        Args:
            image_data_dir (string): Directory with all the images.
            mask_data_dir (string): Directory with all the binary label/annotation.
            list_file: Path to the txt file with: image file name + label/annotation file name + class label.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.n_classes = 1

        case_indexs = []
        image_names = []
        image_mid_names = []
        mask_mid_names = []
        class_vecs = []

        with open(list_file, "r") as f:
            for line in f:
                items = line.split()

                case_index = int(items[0])
                case_indexs.append(case_index)

                image_name = items[2]
                image_name = os.path.join(image_data_dir, image_name)
                image_names.append(image_name)

                image_mid_name = items[3]
                image_mid_name = os.path.join(image_data_dir, image_mid_name)
                image_mid_names.append(image_mid_name)

                mask_mid_name = items[4]
                mask_mid_name = os.path.join(mask_data_dir, mask_mid_name)
                mask_mid_names.append(mask_mid_name)

                class_vec = items[5]
                class_vec = [int(i) for i in class_vec]
                class_vecs.append(class_vec)

        self.case_indexs = case_indexs
        self.image_names = image_names
        self.image_mid_names = image_mid_names
        self.mask_mid_names = mask_mid_names
        self.class_vecs = class_vecs

        self.transform = transform
        self.norm_img = norm_img
        self.norm_cond = norm_cond

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and label and class
        """
        # case index loader
        case_index = self.case_indexs[index]

        # image loader
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('I')

        # image middle slice loader
        image_mid_name = self.image_mid_names[index]
        image_mid = Image.open(image_mid_name).convert('I')

        # label/annotation loader
        mask_mid_name = self.mask_mid_names[index]
        mask_mid = Image.open(mask_mid_name).convert('I')

        # class vector loader
        class_vec = self.class_vecs[index]
        class_vec = torch.FloatTensor(class_vec)

        # get the cropped patch using middle slice
        image_patch, image_mid_patch, mask_mid_patch = get_patch(image, image_mid, mask_mid)
        if self.transform is not None:
            image_patch, image_mid_patch, mask_mid_patch = self.transform(image_patch, image_mid_patch, mask_mid_patch)
            
            image_patch = image_patch.numpy()
            image_mid_patch = image_mid_patch.numpy()
            mask_mid_patch = mask_mid_patch.numpy()
            dis_mid_patch = ndimage.distance_transform_edt(mask_mid_patch)

            image_patch = np.repeat(image_patch, 3, axis=0)
            cond_patch = np.concatenate((image_mid_patch, dis_mid_patch, mask_mid_patch), axis=0)

            image_patch = torch.from_numpy(image_patch).float(); image_patch = self.norm_img(image_patch)
            cond_patch = torch.from_numpy(cond_patch).float(); cond_patch = self.norm_cond(cond_patch)

        return case_index, image_patch, cond_patch, class_vec

    def __len__(self):
        return len(self.image_names)


def get_patch(image, image_mid, mask_mid):
    image = np.array(image)
    image_mid = np.array(image_mid)
    mask_mid = np.array(mask_mid)

    # processing get cropping
    [x,y] = np.where(mask_mid == 1)
    x_size = x.max() - x.min()
    y_size = y.max() - y.min()

    if x_size <= 60 and y_size <=60:  # if tumor<60x60, directly crop 70x70
        x_start = int(x.min() - (70 - x_size)/2); x_end = int(x.max() + (70 - x_size)/2)
        y_start = int(y.min() - (70 - y_size)/2); y_end = int(y.max() + (70 - y_size)/2)

    else:                             # if tumor>60x60, crop the original size add 15% width
        x_start = int(x.min() - 0.15 * x_size); x_end = int(x.max() + 0.15 * x_size)
        y_start = int(y.min() - 0.15 * y_size); y_end = int(y.max() + 0.15 * y_size)

    img_patch = image[x_start:x_end, y_start:y_end]
    img_mid_patch = image_mid[x_start:x_end, y_start:y_end]
    mask_mid_patch = mask_mid[x_start:x_end, y_start:y_end]

    img_patch = Image.fromarray(img_patch)
    img_mid_patch = Image.fromarray(img_mid_patch)
    mask_mid_patch = Image.fromarray(mask_mid_patch)

    return img_patch, img_mid_patch, mask_mid_patch