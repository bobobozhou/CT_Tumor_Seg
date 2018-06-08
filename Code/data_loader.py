import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import ipdb


class CTTumorDataset(Dataset):
    """"CT Tumor 2D Data loader"""

    def __init__(self, image_data_dir, mask_data_dir, edge_data_dir, list_file, transform=None, norm=None):

        """
        Args:
            image_data_dir (string): Directory with all the images.
            mask_data_dir (string): Directory with all the binary label/annotation.
            edge_data_dir (string): Directory with all the boundary/edge
            list_file: Path to the txt file with: image file name + label/annotation file name + class label.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.n_classes = 4
        self.class_name = ['Lung', 'Breast', 'Skin', 'Liver']

        case_indexs = []
        image_names = []
        mask_names = []
        edge_names = []
        class_vecs = []
        with open(list_file, "r") as f:
            for line in f:
                items = line.split()

                case_index = int(items[0])
                case_indexs.append(case_index)

                image_name = items[1]
                image_name = os.path.join(image_data_dir, image_name)
                image_names.append(image_name)

                mask_name = items[2]
                mask_name = os.path.join(mask_data_dir, mask_name)
                mask_names.append(mask_name)

                edge_name = items[3]
                edge_name = os.path.join(edge_data_dir, edge_name)
                edge_names.append(edge_name)

                class_vec = items[4:]
                class_vec = [int(i) for i in class_vec]
                class_vecs.append(class_vec)

        self.case_indexs = case_indexs
        self.image_names = image_names
        self.mask_names = mask_names
        self.edge_names = edge_names
        self.class_vecs = class_vecs
        self.transform = transform
        self.norm = norm

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

        # label/annotation loader
        mask_name = self.mask_names[index]
        mask = Image.open(mask_name).convert('I')

        # edge/boundary loader
        edge_name = self.edge_names[index]
        edge = Image.open(edge_name).convert('I')

        # class vector loader
        class_vec = self.class_vecs[index]
        class_vec = torch.FloatTensor(class_vec)

        if self.transform is not None:
            image = self.transform(image).numpy()
            mask = self.transform(mask).numpy()
            edge = self.transform(edge).numpy()

            image = np.repeat(image, 3, axis=0)
            mask = np.concatenate((mask, 1-mask), axis=0)
            edge = np.concatenate((edge, 1-edge), axis=0)

            image = torch.from_numpy(image).float(); image = self.norm(image)
            mask = torch.from_numpy(mask)
            edge = torch.from_numpy(edge)

        return case_index, image, mask, edge, class_vec

    def __len__(self):
        return len(self.image_names)
