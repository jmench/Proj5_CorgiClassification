# Author: Jordan Menchen

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from dataset import read_train_sets

def main():
    train_set = './data/training_data'
    test_set = './data/testing_data'
    image_size = 3
    classes = ['pembroke', 'cardigan']
    validation = 0.2

    datasets = read_train_sets(train_set, image_size, classes, validation)
    print(datasets.train.num_examples())

if __name__ == "__main__": main()