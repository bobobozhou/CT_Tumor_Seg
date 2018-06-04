import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import ipdb


class CTTumorDataset(Dataset):
    """"CT Tumor 2D Data loader"""

    def __init__(self, image_data_dir, label_data_dir, edge_data_dir, list_file, transform=None, norm=None):

        """
        Args:
            image_data_dir (string): Directory with all the images.
            label_data_dir (string): Directory with all the binary label/annotation.
            edge_data_dir (string): Directory with all the boundary/edge
            list_file: Path to the txt file with: image file name + label/annotation file name + class label.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.n_classes = 4
        self.class_name = ['Lung', 'Breast', 'Skin', 'Liver']

        image_names = []
        mask_names = []
        edge_names = []
        class_vecs = []
        with open(list_file, "r") as f:
            for line in f:
                items = line.split()

                image_name = items[0]
                image_name = os.path.join(image_data_dir, image_name)
                image_names.append(image_name)

                mask_name = items[1]
                mask_name = os.path.join(label_data_dir, mask_name)
                mask_names.append(mask_name)

                edge_name = items[2]
                edge_name = os.path.join(edge_data_dir, edge_name)
                edge_names.append(edge_name)

                class_vec = items[3:]
                class_vec = [int(i) for i in class_vec]
                class_vecs.append(class_vec)

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
        # image loader
        image_name = self.image_names[index]
        image = np.array(Image.open(image_name))

        # label/annotation loader
        mask_name = self.mask_names[index]
        mask = np.array(Image.open(mask_name))

        # edge/boundary loader
        edge_name = self.edge_names[index]
        edge = np.array(Image.open(edge_name))

        # class vector loader
        class_vec = self.class_vecs[index]
        ipdb.set_trace()

        if self.transform is not None:
            image = self.transform(image); image = self.norm(image)
            mask = self.transform(mask)
            edge = self.transform(edge)

        return image, mask, edge, torch.FloatTensor(class_vec)

    def __len__(self):
        return len(self.image_names)
