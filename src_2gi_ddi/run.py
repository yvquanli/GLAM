import argparse
from model import Model
from dataset import Dataset
from utils import seed_torch, auto_dataset, model_args

import warnings; warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_root', default='../../Dataset/GLAM-DDI', type=str, help='dataset root path')
parser.add_argument('--dataset', type=str, default='drugbank_caster', help='drugbank_caster')
parser.add_argument('--split', type=str, default='random', help='random, scaffold')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--split_seed', type=int, default=1234)
parser.add_argument('--gpu', default=0, type=int, help='cuda device id')
parser.add_argument('--note', default='normal_run32rfds', type=str, help='you can write note here')

parser.add_argument('--hid_dim_alpha', default=4, type=int, help='hidden size of model')
parser.add_argument('--mol_block', type=str, default='_TripletMessage', help='_TripletMessage, _NNConv')
parser.add_argument('--e_dim', default=1024, type=int, help='output size of model')
parser.add_argument('--out_dim', default=1, type=int, help='output size of model')
parser.add_argument('--message_steps', default=3, type=int)
parser.add_argument('--mol_readout', default='GlobalPool5', type=str, help='GlobalPool5, Set2Set, GlobalLAPool')

parser.add_argument('--pre_norm', default='_None', type=str)
parser.add_argument('--graph_norm', default='_PairNorm', type=str)
parser.add_argument('--flat_norm', default='_None', type=str)
parser.add_argument('--end_norm', default='_None', type=str)
parser.add_argument('--pre_do', default='_None()', type=str)
parser.add_argument('--graph_do', default='_None()', type=str)
parser.add_argument('--flat_do', default='Dropout(0.2)', type=str)
parser.add_argument('--end_do', default='Dropout(0.2)', type=str)
parser.add_argument('--pre_act', default='RReLU', type=str)
parser.add_argument('--graph_act', default='RReLU', type=str)
parser.add_argument('--flat_act', default='RReLU', type=str)
parser.add_argument('--end_act', default='RReLU', type=str)
parser.add_argument('--graph_res', default=1, type=int)

parser.add_argument('--batch_size', default=50, type=int, help='number of batch_size')
parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--loss', default='bcel', type=str, help='ce,wce,focal,bfocal...')
parser.add_argument('--optim', default='Adam', type=str, help='range, adam, sgd')
parser.add_argument('--k', default=6, type=int, help='lookahead steps')  # for ranger optimization only
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--lr_reduce_rate', default=0.7, type=float)
parser.add_argument('--lr_reduce_patience', default=20, type=int)
parser.add_argument('--early_stop_patience', default=50, type=int)
parser.add_argument('--verbose_patience', default=100, type=int)

args = parser.parse_args()
seed_torch(args.seed)

print('Loading dataset...')
args, dataset, Trainer = auto_dataset(args)
train_dataset, valid_dataset, test_dataset = dataset.train, dataset.val, dataset.test

print('Training init...')
model = Model(dataset.mol_num_node_features, dataset.mol_num_edge_features, **model_args(args))

trainer = Trainer(args, model, train_dataset, valid_dataset, test_dataset)
trainer.train_and_test()
