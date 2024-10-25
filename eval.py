import numpy as np
import torch
import random
import os

from argSettings import arg
from train import ModelTrainer
from DatasetChoose.data_utils import dataset_builder
from Model.backbone import EmbeddingImagenet, ResNet
from Model.model import Gnn


if __name__ == '__main__':
    arg.test_model = 'D-cifar_N-10_K-5_B-20_T-True_P-support_SEED-222_Backbone-conv' if arg.test_model is None else arg.test_model

    list1 = arg.test_model.split('_')
    param = {}

    for i in range(len(list1)):
        param[list1[i].split('-',1)[0]] = list1[i].split('-',1)[1]
    arg.dataset = param['D']
    arg.num_ways = int(param['N'])
    arg.num_shots = int(param['K'])
    arg.train_batch_size = int(param['B'])
    arg.transductive = True if param['T'] else False


    np.random.seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    random.seed(arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    arg.log_dir = 'asset/log'
    arg.log_dir_user = arg.log_dir if arg.log_dir_user is None else arg.log_dir_user
    arg.log_dir = arg.log_dir_user
    # cross way test
    num_queries = arg.num_ways

    arg.num_supports = arg.num_ways * arg.num_shots

    # log path setting
    if not os.path.exists('asset/checkpoints'):
        os.makedirs('asset/checkpoints')
    if not os.path.exists('asset/checkpoints/' + arg.exp_name):
        os.makedirs('asset/checkpoints/' + arg.exp_name)

    # backbone setting
    if arg.backbone == 'conv':
        enc_module = EmbeddingImagenet(emb_size=arg.emb_size)
    elif arg.backbone == 'res':
        enc_module = ResNet(arg.emb_size)
    else:
        print('wrong backbone settings!!!')
        raise NameError('wrong backbone settings!!!')

    # num_shots and trans setting
    if arg.num_shots in [1, 5, 10]:
        if arg.pool_mode in ['support', 'kn']:
            arg.ks = [0.6, 0.5]
        else:
            print('wrong mode setting!!!')
            raise NameError('wrong mode setting!!!')
    else:
        print('wrong shot and T settings!!!')
        raise NameError('wrong shot and T settings!!!')

    if not arg.transductive:
        # trans = false
        gnn_module = Gnn(arg.ks, arg.in_dim, arg.num_ways, 1)
    else:
        gnn_module = Gnn(arg.ks, arg.in_dim, arg.num_ways, num_queries)

    DatasetLoader = dataset_builder(arg.dataset)
    test_loader = DatasetLoader(root=arg.dataset_root, partition='test')

    data_loader = {'test': test_loader}

    tester = ModelTrainer(enc_module=enc_module,
                           gnn_module=gnn_module,
                           data_loader=data_loader,
                           )

    checkpoint = torch.load('asset/checkpoints/{}/'.format(arg.exp_name) +  'model_best.pth.tar', map_location=arg.device )

    tester.enc_module.load_state_dict(checkpoint['enc_module_state_dict'])
    print("load pre-trained enc_nn done!")

    tester.gnn_module.load_state_dict(checkpoint['gnn_module_state_dict'])
    print("load pre-trained gnn_nn done!")

    tester.val_acc = checkpoint['val_acc']
    tester.global_step = checkpoint['iteration']
    print(tester.global_step, tester.val_acc)

    tester.eval(partition='test',num_ways=5)









