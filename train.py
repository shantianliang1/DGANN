import os.path
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DatasetChoose.data_utils import dataset_builder
from Model.backbone import EmbeddingImagenet, ResNet
from Model.model import Gnn
from argSettings import arg
from logger import log_scalar, log_step, log


def save_checkpoint(state, best_flag, exp_name):
    torch.save(state, 'asset/checkpoints/{}/'.format(exp_name) + 'checkpoint.pth.tar')
    if best_flag:
        shutil.copyfile('asset/checkpoints/{}/'.format(exp_name) + 'checkpoint.pth.tar',
                        'asset/checkpoints/{}/'.format(exp_name) + 'model_best.pth.tar')


def label2edge(label):
    num_samples = label.size(1)

    # reshape
    label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
    label_j = label_i.transpose(1, 2)

    # compute edge
    edge = torch.eq(label_i, label_j).float().to(arg.device)
    return edge


def one_hot_encode(num_classes, class_idx):
    return torch.eye(num_classes)[class_idx].to(arg.device)


def adjust_learning_rate(optimizers, lr, it):
    lr = lr * (0.5 ** (int(it / arg.dec_lr)))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



class ModelTrainer(object):
    def __init__(self, enc_module, gnn_module, data_loader):
        # init settings
        self.enc_module = enc_module.to(arg.device)
        self.gnn_module = gnn_module.to(arg.device)
        self.data_loader = data_loader

        # set module parameters
        self.module_parameters = list(self.enc_module.parameters()) + list(self.gnn_module.parameters())

        # set optimizer
        # Adam 优化算法 梯度下降优化
        self.optimizer = optim.Adam(params=self.module_parameters,
                                    lr=arg.lr,
                                    weight_decay=arg.weight_decay)

        # loss
        # NLLLoss 损失函数
        self.node_loss = nn.NLLLoss()

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

    def train(self):
        val_acc = self.val_acc

        # distinguish support and query
        num_ways_train = arg.num_ways
        num_shots_train = arg.num_shots


        num_supports = num_ways_train * num_shots_train
        num_queries = num_ways_train * 1
        num_samples = num_supports + num_queries

        # 掩码test
        support_edge_mask = torch.zeros(arg.train_batch_size, num_samples, num_samples).to(arg.device)
        support_edge_mask[:, num_supports:, :] = 0
        query_edge_mask = torch.ones(arg.train_batch_size, num_samples, num_samples).to(arg.device)
        query_edge_mask[:, :num_supports, :] = 0
        evaluation_mask = torch.ones(arg.train_batch_size, num_samples).to(arg.device)


        # it = iteration
        for it in range(self.global_step + 1, arg.train_iteration + 1):
            # init grad
            self.optimizer.zero_grad()
            self.global_step = it

            # load task data list
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader['train'].get_task_batch(num_tasks=arg.train_batch_size,
                                                                     num_ways=  num_ways_train,
                                                                     num_shots= num_shots_train,
                                                                     seed=it + arg.seed
                                                                     )

            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = label2edge(full_label)

            # init edge setting
            init_edge = full_edge.clone()
            init_edge[:, num_supports:, :] = 0.5
            init_edge[:, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, num_supports + i, num_supports + i] = 1.0

            self.enc_module.train()
            self.gnn_module.train()

            # 1.data
            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1)

            one_hot_label = one_hot_encode(num_ways_train, support_label.long())
            query_padding = (1 / num_ways_train) * torch.ones(
                [full_data.shape[0]] + [num_queries] + [num_ways_train],
                device=one_hot_label.device)
            one_hot_label = torch.cat([one_hot_label, query_padding], dim=1)

            full_data = torch.cat([full_data, one_hot_label], dim=-1)

            # check trans and make different settings
            if arg.transductive:
                full_node_out = self.gnn_module(init_edge, full_data)
            else:
                evaluation_mask[:, num_supports:] = 0

                # batch_size * num_support * feat-dim
                support_data = full_data[:, :num_supports]
                # batch_size * num_query * feat-dim
                query_data = full_data[:, num_supports:]
                # batch_size * num_queries * num_support * feat-dim
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1, 1)
                # (batch_size * num_queries) * num_support * feat-dim
                support_data_tiled = support_data_tiled.view(arg.train_batch_size * num_queries, num_supports, -1)
                # (batch_size * num_queries) * 1 * feat-dim
                query_data_reshaped = query_data.contiguous().view(arg.train_batch_size * num_queries,
                                                                   -1).unsqueeze(1)
                # (batch_size * num_queries) * (num_support + 1) * feat-dim
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped], 1)
                # batch_size * (num_support +1) * (num_support +1)
                input_edge_feat = 0.5 * torch.ones(arg.train_batch_size, num_supports + 1, num_supports + 1).to(
                    arg.device)
                # batch_size * (num_support +1) * (num_support +1)
                input_edge_feat[:, :num_supports, :num_supports] = init_edge[:, :num_supports, : num_supports]
                # (batch_size * num_queries) * (num_support + 1) * (num_support +1)
                input_edge_feat = input_edge_feat.repeat(num_queries, 1, 1)

                # 2.gnn
                # (batch_size* num_queries) * (num_support +1 ) * num_class
                node_out = self.gnn_module(input_edge_feat, input_node_feat)
                # batch_size * num_queries * (num_support + 1) * num_class
                node_out = node_out.view(arg.train_batch_size, num_queries, num_supports + 1, arg.num_ways)
                full_node_out = torch.zeros(arg.train_batch_size, num_samples, arg.num_ways).to(
                    arg.device)
                full_node_out[:, :num_supports, :] = node_out[:, :, :num_supports, :].mean(1)
                full_node_out[:, num_supports:, :] = node_out[:, :, num_supports:, :].squeeze(2)

            # 3.loss
            query_node_out = full_node_out[:, num_supports:]
            full_label = full_label.to(query_node_out.device)

            node_pre = torch.argmax(query_node_out, dim=-1)

            # 掩码test
            node_acc = torch.sum((node_pre == full_label[:, num_supports:].long()) * evaluation_mask[:,
                                                        num_supports:]).float()/(num_queries * arg.train_batch_size)
            node_loss = [self.node_loss(data.squeeze(1), label.squeeze(1).long()) * evaluation_mask[:, num_supports:] \
                         for data, label in zip(query_node_out.chunk(query_node_out.size(1), dim=1),
                                                full_label[:, num_supports:].chunk(full_label[:, num_supports:].size(1),
                                                                                   dim=1))]
            node_loss = torch.stack(node_loss, dim=0)
            valid_node_loss = node_loss.mean()

            valid_node_loss.backward()

            self.optimizer.step()

            adjust_learning_rate(optimizers=[self.optimizer], lr=arg.lr, it=self.global_step)

            # logging
            log_scalar('train/loss', valid_node_loss.item(), self.global_step)  # Convert to scalar
            log_scalar('train/node_acc', node_acc.item(), self.global_step)  # Convert to scalar


            # evaluation
            if self.global_step % arg.test_interval == 0:
                val_acc = self.eval(partition='val')
                best_flag = 0
                if val_acc > self.val_acc:
                    self.val_acc = val_acc
                    best_flag = 1

                log_scalar('val/best_acc', self.val_acc.item(), self.global_step)

                #log_scalar('val/best_acc', self.val_acc, self.global_step)

                save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'gnn_module_state_dict': self.gnn_module.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, best_flag, arg.exp_name)

            log_step(global_step=self.global_step)


    def eval(self, partition='test', log_flag=True, num_ways=None):
        if num_ways is None:
            num_ways = arg.num_ways
        best_acc = 0

        print('num_ways_test:', num_ways)
        num_shots_test = arg.num_shots

        num_supports = num_ways * num_shots_test
        num_queries = num_ways * 1
        num_samples = num_supports + num_queries

        support_edge_mask = torch.zeros(arg.train_batch_size, num_samples, num_samples).to(arg.device)
        support_edge_mask[:, num_supports:, :] = 0
        query_edge_mask = torch.ones(arg.train_batch_size, num_samples, num_samples).to(arg.device)
        query_edge_mask[:, :num_supports, :] = 0
        evaluation_mask = torch.ones(arg.train_batch_size, num_samples).to(arg.device)

        query_node_acc = []

        for it in range(arg.test_iteration // arg.test_batch_size):
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader[partition].get_task_batch(num_tasks=arg.test_batch_size,
                                                                       num_ways=num_ways,
                                                                       num_shots=num_shots_test,
                                                                       seed=it)

            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = label2edge(full_label)

            init_edge = full_edge.clone()
            init_edge[:, num_supports:, :] = 0.5
            init_edge[:, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, num_supports + i, num_supports + i] = 1.0

            self.enc_module.eval()
            self.gnn_module.eval()

            # 1. data
            full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1)

            one_hot_label = one_hot_encode(num_ways, support_label.long())
            query_padding = (1 / num_ways) * torch.ones([full_data.shape[0]] + [num_queries] + [num_ways],
                                                            device=one_hot_label.device)

            one_hot_label = torch.cat([one_hot_label, query_padding], dim=1)

            full_data = torch.cat([full_data, one_hot_label], dim=-1)

            if arg.transductive:
                full_node_out = self.gnn_module(init_edge, full_data)
            else:
                evaluation_mask[:, num_supports:] = 0

                support_data = full_data[:, :num_supports]
                query_data = full_data[:, num_supports:]
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1, 1)
                support_data_tiled = support_data_tiled.view(arg.test_batch_size * num_queries, num_supports, -1)
                query_data_reshaped = query_data.contiguous().view(arg.test_batch_size * num_queries, -1).unsqueeze(1)
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped], 1)
                input_edge_feat = 0.5 * torch.ones(arg.test_batch_size, num_supports + 1, num_supports + 1).to(
                    arg.device)
                input_edge_feat[:, :num_supports, :num_supports] = init_edge[:, :num_supports, :num_supports]
                input_edge_feat = input_edge_feat.repeat(num_queries, 1, 1)

                #  2. gnn
                node_out = self.gnn_module(input_edge_feat, input_node_feat)
                node_out = node_out.view(arg.test_batch_size, num_queries, num_supports + 1, arg.num_ways)
                full_node_out = torch.zeros(arg.test_batch_size, num_samples, arg.num_ways).to(arg.device)
                full_node_out[:, :num_supports, :] = node_out[:, :, :num_supports, :].mean(1)
                full_node_out[:, num_supports:, :] = node_out[:, :, num_supports:, :].squeeze(2)

            # 3. loss
            query_node_out = full_node_out[:, num_supports:]
            full_label = full_label.to(query_node_out.device)

            node_pre = torch.argmax(query_node_out, dim=1)
            node_acc = torch.sum(torch.eq(node_pre, full_label[:, num_supports:].long())).float() \
                / node_pre.size(0) / num_queries
            query_node_acc += [node_acc.item()]

        # logging
        if log_flag:
            log('--------------------------')
            log_scalar('{}/node_acc'.format(partition), np.array(query_node_acc).mean(), self.global_step)
            log('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                (it,
                 np.array(query_node_acc).mean() * 100,
                 np.array(query_node_acc).std() * 100,
                 1.96 * np.array(query_node_acc).std() / np.sqrt(
                     float(len(np.array(query_node_acc)))) * 100))
            log('--------------------------')


        return np.array(query_node_acc).mean()






if __name__ == '__main__':

    # seed settings
    np.random.seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    random.seed(arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    arg.log_dir = 'asset/log'
    arg.log_dir_user = arg.log_dir if arg.log_dir_user is None else arg.log_dir_user
    arg.log_dir = arg.log_dir_user

    num_queries = arg.num_ways

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
    train_loader = DatasetLoader(root=arg.dataset_root, partition='train')
    valid_loader = DatasetLoader(root=arg.dataset_root, partition='val')

    data_loader = {'train': train_loader,
                   'val': valid_loader}

    trainer = ModelTrainer(enc_module=enc_module,
                           gnn_module=gnn_module,
                           data_loader=data_loader,
                           )

    trainer.train()

