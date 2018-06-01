import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import ipdb


class CTTumorDataset(Dataset):
    """"CT Tumor 2D Data loader"""

    def __init__(self, image_data_dir, label_data_dir, edge_data_dir, list_file, transform=None):

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
        label_names = []
        edge_names = []
        class_vecs = []
        with open(list_file, "r") as f:
            for line in f:
                items = line.split()

                image_name = items[0]
                image_name = os.path.join(image_data_dir, image_name)
                image_names.append(image_name)

                label_name = items[1]
                label_name = os.path.join(label_data_dir, label_name)
                label_names.append(label_name)

                edge_name = items[3]
                edge_name = os.path.join(edge_data_dir, edge_name)
                edge_names.append(edge_name)

                class_vec = items[3:]
                class_vec = [int(i) for i in class_vec]
                class_vecs.append(class_vec)

        self.image_names = image_names
        self.label_names = label_names
        self.edge_names = edge_names
        self.class_vecs = class_vecs
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and label and class
        """
        # image loader
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')

        # label/annotation loader
        label_name = self.label_names[index]
        label = Image.open(label_name)

        # edge/boundary loader
        edge_name = self.edge_names[index]
        edge = Image.open(edge_name)

        # class vector loader
        class_vec = self.class_vecs[index]

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
            edge = self.transform(edge)

        return image, label, edge, torch.FloatTensor(class_vec)

    def __len__(self):
        return len(self.image_names)
