import random

import torch
from brotli import decompress
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import pickle
import tqdm
import cv2
from argSettings import arg



class TieredImagenetLoader(Dataset):
    def __init__(self, root, partition='train' ):
        self.root = root
        self.partition = partition
        self.data_size = [3,84,84]


        self.data = self.load_data_pickle()

    def load_data_pickle(self):
        print("Loading dataset")

        labels_name = '{}/tiered-imagenet/{}_labels.pkl'.format(self.root, self.partition)
        images_name = '{}/tiered-imagenet/{}_images.npz'.format(self.root, self.partition)

        print('labels:', labels_name)
        print('images:', images_name)


        if not os.path.exists(images_name):
            png_pkl = images_name[:-4] + '_png.pkl'
            if os.path.exists(png_pkl):
                decompress(images_name, png_pkl)
            else:
                raise ValueError('path png_pkl not exits')

        if os.path.exists(images_name) and os.path.exists(labels_name):
            try:
                with open(labels_name) as f:
                    data = pickle.load(f)
                    label_specific = data["label_specific"]
            except:
                with open(labels_name, 'rb') as f:
                    data = pickle.load(f, encoding = 'bytes')
                    label_specific = data['label_specific']
            print('read label data:{}'.format(len(label_specific)))
        labels = label_specific

        with np.load(images_name, mmap_mode="r", encoding='latin1') as data:
            image_data = data["images"]
            print('read image data:{}'.format(image_data.shape))

        data = {}
        n_classes = np.max(labels) + 1
        for c_idx in range(n_classes):
            data[c_idx] = []
            idxs = np.where(labels == c_idx)[0]
            np.random.RandomState(arg.seed).shuffle(idxs)
            for i in idxs:
                image2resize = Image.fromarray(np.uint8(image_data[i, :, :, :]))
                image_resized = image2resize.resize((self.data_size[2], self.data_size[1]))
                image_resized = np.array(image_resized, dtype='float32')

                # Normalize
                image_resized = np.transpose(image_resized, (2, 0, 1))
                image_resized[0, :, :] -= 120.45 # R
                image_resized[1, :, :] -= 115.74 # G
                image_resized[2, :, :] -= 104.65 # B
                image_resized /= 127.5
                data[c_idx].append(image_resized)
        return data

    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=5,
                       num_shots=1,
                       num_queries=1,
                       seed=None):
        if seed is not None:
            random.seed(seed)

        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                            dtype='float32')
            query_data.append(data)
            query_label.append(label)

        full_class_list = list(self.data.keys())

        for t_idx in range(num_tasks):
            task_class_list = random.sample(full_class_list, num_ways)

            for c_idx in range(num_ways):
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)

                for i_idx in range(num_shots):
                    support_data[i_idx + c_idx * num_shots][t_idx] = class_data_list[i_idx]
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = class_data_list[num_shots + i_idx]
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        support_data = torch.stack([torch.from_numpy(data).float().to(arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]


def decompress(path, output):
    with open(output, 'rb') as f:
        array = pickle.load(f, encoding='bytes')
    images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
    for ii, item in tqdm(enumerate(array), desc='decompress'):
        im = cv2.imdecode(item, 1)
        images[ii] = im
    np.savez(path, images=images)