import argparse

def set_exp_name():
    exp_name = 'D-{}'.format(arg.dataset)
    exp_name += '_N-{}_K-{}'.format(arg.num_ways, arg.num_shots)
    exp_name += '_B-{}_T-{}'.format(arg.train_batch_size, arg.transductive)
    exp_name += '_P-{}'.format(arg.pool_mode)
    exp_name += '_SEED-{}'.format(arg.seed)
    exp_name += '_Backbone-{}'.format(arg.backbone)

    return exp_name


parser = argparse.ArgumentParser()

# device and root related
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--dataset_root', type=str, default='./dataset')
parser.add_argument('--dataset', type=str, default='cifar', help='mini/tiered/cifar')
parser.add_argument('--display_step', type=int, default=100)
parser.add_argument('--log_step', type=int, default=100)

# train related
parser.add_argument('--num_ways', type=int, default=10)
parser.add_argument('--num_shots', type=int, default=5)
#parser.add_argument('--num_queries', type=int, default=5, help='5/10 ')
#parser.add_argument('--num_supports', type=int, default=None, help='num_ways * num_shots')
parser.add_argument('--train_iteration', type=int, default=100000, help='tiered=200000')
parser.add_argument('--train_batch_size', type=int, default=20)
parser.add_argument('--seed', type=int, default=222)

# model related
parser.add_argument('--backbone', type=str, default='conv', help='conv/res')
parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--pool_mode', type=str, default='support', help='support/kn')
parser.add_argument('--transductive', type=bool, default=True)  # the label of using transductive

# test parameters
parser.add_argument('--test_iteration', type=int, default=1000)
parser.add_argument('--test_interval', type=int, default=1000)
parser.add_argument('--test_batch_size', type=int, default=10)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--grad_clip', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--dec_lr', type=int, default=15000)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
parser.add_argument('--log_dir_user', type=str, default=None, help='User-defined log directory')
parser.add_argument('--log_interval', type=str, default=None)
parser.add_argument('--log_file', type=str, default=None)
parser.add_argument('--ks', type=list, default=[])
parser.add_argument('--test_model', type=str, default=None)

arg = parser.parse_args()
arg.exp_name = set_exp_name() if arg.exp_name is None else arg.exp_name
arg.in_dim = arg.emb_size + arg.num_ways
print(set_exp_name())