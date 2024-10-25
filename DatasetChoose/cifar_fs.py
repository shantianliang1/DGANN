import pickle
import random
import numpy as np
import os
import torch
from argSettings import arg
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from DataAugmentation.image_augmentation import ImageAugmentation


# cifar-fs
class CifarFsLoader(Dataset):
    def __init__(self, root, partition='train'):
        super(CifarFsLoader, self).__init__()
        self.root = root
        self.partition = partition
        self.data_size = [3, 32, 32]

        mean_pix = [x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
        std_pix = [x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
        normalize = transforms.Normalize(mean_pix, std_pix)

        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                 # lambda x: ImageAugmentation().random_aug(np.asarray(x)),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        else:
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        self.data = self.load_dataset()

    def load_dataset(self):
        dataset_path = os.path.join(self.root, 'cifar-fs', 'compacted_datasets','cifar_fs_%s.pickle' % self.partition)
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)

        for c_idx in data:
            for i_idx in range(len(data[c_idx])):
                # resize
                # width * height = 32 * 32
                image_data = Image.fromarray(np.uint8(data[c_idx][i_idx]))
                image_data = image_data.resize((self.data_size[2], self.data_size[1]))
                data[c_idx][i_idx] = image_data
        return data


    def get_task_batch(self, num_tasks=5, num_ways=20, num_shots=1, num_queries=1, seed=None):
        if seed is not None:
            random.seed(seed)

        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size, dtype='float32')
            label = np.zeros(shape=[num_tasks], dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] +self.data_size, dtype='float32')
            label = np.zeros(shape=[num_tasks], dtype='float32')
            query_data.append(data)
            query_label.append(label)

        full_class_list = list(self.data.keys())

        for t_idx in range(num_tasks):
            task_class_list = random.sample(full_class_list, num_ways)

            for c_idx in range(num_ways):
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)

                for i_idx in range(num_shots):
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        support_data = torch.stack([torch.from_numpy(data).float().to(arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]









