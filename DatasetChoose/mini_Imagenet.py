import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import pickle
from argSettings import arg
from DataAugmentation.image_augmentation import ImageAugmentation


# mini-Imagenet
class MiniImagenetLoader(Dataset):
    def __init__(self, root, partition='train'):
        super(MiniImagenetLoader, self).__init__()
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]

        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean_pix, std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                # lambda x: ImageAugmentation().random_aug(np.asarray(x)),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        self.data = self.load_dataset()

        # dataset loader

    def load_dataset(self):

        # from pickle
        dataset_path = os.path.join(self.root, 'mini-imagenet/compacted_datasets',
                                    'mini_imagenet_%s.pickle' % self.partition)
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)

        # c_idx = class_index  i_idx = image_index
        for c_idx in data:
            for i_idx in range(len(data[c_idx])):
                # resize
                # width * height = 84 * 84
                image_data = Image.fromarray(np.uint8(data[c_idx][i_idx]))
                image_data = image_data.resize((self.data_size[2], self.data_size[1]))
                data[c_idx][i_idx] = image_data
        return data

        # task for model train

    def get_task_batch(self, num_tasks=5, num_ways=20, num_shots=1, num_queries=1, seed=None):
        if seed is not None:
            random.seed(seed)

        # initialize data and label as []
        support_data, support_label, query_data, query_label = ([] for _ in range(4))
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size, dtype='float32')
            label = np.zeros(shape=[num_tasks], dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size, dtype='float32')
            label = np.zeros(shape=[num_tasks], dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get class list
        full_class_list = list(self.data.keys())

        # t_idx = task_index
        for t_idx in range(num_tasks):
            task_class_list = random.sample(full_class_list, num_ways)

            # each class in task
            for c_idx in range(num_ways):
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)
                # support set
                for i_idx in range(num_shots):
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        support_data = torch.stack([torch.from_numpy(data).float().to(arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]
