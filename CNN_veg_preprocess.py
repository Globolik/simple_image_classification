import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import Image

import numpy as np

import os
import shutil

from ast import literal_eval

import pandas as pd

# train (15000 images)
# test (3000 images)
# validation (3000 images)
train_data_path = '/home/globolik/simple_torch_nn/vegetables/Vegetable Images/train'
val_data_path = '/home/globolik/simple_torch_nn/vegetables/Vegetable Images/validation'
test_data_path = '/home/globolik/simple_torch_nn/vegetables/Vegetable Images/test'
def check_data(data_path, mode):



    # arr with shape 1,3 with random value that we will drop
    arr_of_shapes = np.ndarray(shape=(1,3), dtype=int)
    class_folders = os.listdir(data_path)
    image_paths_different_size = []
    for class_folder_name in class_folders:

        class_folder_path = data_path+'/'+class_folder_name
        img_names = os.listdir(class_folder_path)

        for img_name in img_names:
            image_path = class_folder_path+'/'+img_name

            img = Image.open(image_path)
            shape = [[*np.array(img).shape]]
            if shape!=[[224, 224, 3]]:
                image_paths_different_size.append(image_path)
            arr_of_shapes = np.concatenate((arr_of_shapes, shape), axis=0)

    # drop first value that numpy inited array with
    arr_of_shapes = np.delete(arr_of_shapes, 0, axis=0)


    print(f'{mode} images with different size {image_paths_different_size}')
    with open(f'diff_img_{mode}.txt', 'w') as f:
        f.write(str(image_paths_different_size))
    # out:
    # images with different size [
    # '/home/globolik/simple_torch_nn/vegetables/Vegetable Images/train/Papaya/0176.jpg',
    # '/home/globolik/simple_torch_nn/vegetables/Vegetable Images/train/Papaya/0126.jpg',
    # '/home/globolik/simple_torch_nn/vegetables/Vegetable Images/train/Papaya/0741.jpg',
    # '/home/globolik/simple_torch_nn/vegetables/Vegetable Images/train/Bitter_Gourd/0609.jpg',
    # '/home/globolik/simple_torch_nn/vegetables/Vegetable Images/train/Bitter_Gourd/0430.jpg',
    # '/home/globolik/simple_torch_nn/vegetables/Vegetable Images/train/Bitter_Gourd/0526.jpg'
    # ]
    print(np.unique(arr_of_shapes, return_counts=True, axis=0))
    # out:
    # (array([[193, 224, 3],
    #         [198, 224, 3],
    #         [200, 224, 3],
    #         [205, 224, 3],
    #         [210, 224, 3],
    #         [211, 224, 3],
    #         [224, 224, 3]]), array([1, 1, 1, 1, 1, 1, 14994]))

# check_data(test_data_path, mode='test')



def move_images_with_diff_size(mode):
    with open(f'diff_img_{mode}.txt', 'r') as f:
        list_of_images = literal_eval(f.read())

        try:
            os.mkdir('diff_img/')
        except:
            pass

        for img_path in list_of_images:
           img_name = img_path.split('/')[-1]
           shutil.move(img_path, f'diff_img/{img_name}')

check_data(test_data_path, mode='test')
move_images_with_diff_size(mode='test')

def create_annotations(train=True):
    if train:
        path = train_data_path
        mode = 'train'
    else:
        path = test_data_path
        mode = 'test'
    class_folders = os.listdir(path)
    data_annotations = pd.DataFrame(columns=['name', 'label'])
    decode_annotations = []
    for label ,class_folder_name in enumerate(class_folders):

        decode_annotations.append([label, class_folder_name])
        class_folder_path = path + '/' + class_folder_name
        img_names = os.listdir(class_folder_path)

        for img_name in img_names:
            img_name = class_folder_name + "/" + img_name
            data_annotations = data_annotations.append(
                {'name': img_name, 'label': label},
                ignore_index=True,
            )

    print(data_annotations)
    data_annotations.to_csv(f'annotations_{mode}.csv', header=False, index=False)

create_annotations(train=False)


def calculate_mean_std():
    # arr with shape 1,3 with random value that we will drop
    class_folders = os.listdir(train_data_path)
    std_list = [[], [], []]
    mean_list = [[], [], []]
    for class_folder_name in class_folders:

        class_folder_path = train_data_path + '/' + class_folder_name
        img_names = os.listdir(class_folder_path)

        for img_name in img_names:
            image_path = class_folder_path + '/' + img_name

            img = Image.open(image_path)
            arr = np.transpose(np.array(img))
            for chanel in range(3):
                mean = arr[chanel].flatten().mean()
                std = arr[chanel].flatten().std()
                mean_list[chanel].append(mean)
                std_list[chanel].append(std)
    mean = np.array(mean_list, dtype=int).mean(axis=1,dtype=int).tolist()
    std = np.array(std_list, dtype=int).mean(axis=1, dtype=int).tolist()
    with open('mean_std.txt', 'w') as f:
        f.write(f'{dict(mean= mean, std= std)}')



# calculate_mean_std()
