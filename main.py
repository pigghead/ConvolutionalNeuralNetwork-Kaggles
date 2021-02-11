"""
AUTHOR: Justin Armstrong
EMAIL: jxa1762@rit.edu

SUMMARY: This is an exploration of the convolutional network. A CNN accepts 2D and 3D data
-- Typical use case is 2D
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

REBUILD_DATA = False  # only needs to be set to true once

# The same general steps for image processing are taken; below is basic data setup
class DogsVSCats():
    IMG_SIZE = 50  # image size needs to be uniform for input

    # Point at the directories for the dog and cat folders
    DOGS = "PetImages/Dog"
    CATS = "PetImages/Cat"

    LABELS = {CATS: 0, DOGS: 1}  # To label our dataset; convert to 1-hot in future
    training_data = []  # populate w/ img of cats & dogs & their labels

    # Keeping count
    cat_count = 0
    dog_count = 0

    def make_training_data (self):
        # Looks into our PetImages directory
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):  # os.listdir creates a list of directory items
                try:
                    # Create a path to each item in the directory
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # change the IMG to gray
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))  # resize (50 x 50)
                    # np.eye() creates a 1Hot array of the given size, indexer determines
                    # which value is 1
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.cat_count += 1
                    elif label == self.DOGS:
                        self.dog_count += 1

                except Exception as e:
                    #print(str(e))
                    pass

        # Shuffle the data, output the file to work with
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats: ", self.cat_count)
        print("Dogs: ", self.dog_count)


if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)

#print(training_data[2])
plt.imshow(training_data[0][0], cmap="gray")
plt.show()

# Create the model itself
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)
