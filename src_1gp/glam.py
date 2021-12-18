import argparse
from pathlib import Path
import time
import subprocess
from random import choice
from utils import GPUManager, config2cmd, md5
from trainer import Inferencer
from dataset import Dataset, PertubationDataset


class GLAM():
    def __init__(self, args):
        self.args = args
        self.gm = GPUManager()
        self.seeds = [12, 123, 1234, 16, 32, 50, 64, 100, 128, 200]
        self.start = time.time()
        self.logs_dir = Path('./log_{}/'.format(args.dataset))
        Path.mkdir(self.logs_dir, exist_ok=True)
        self.inferencer = Inferencer(args.dataset)
        self.searched_params = []

        self.log('Solver for {} running start @ {}'.format(args.dataset, time.asctime(time.localtime(self.start))))
        # assert len(self.gm.gpus) > 0
        self.log('{} gpus available'.format(len(self.gm.gpus)))

    def random_search(self):
        proc = []
        for i in range(self.args.n_init_configs):
            config, config_id = self.generate_config()  # # take a point out of the search space
            self.log('Configuration {} start: \n config_id is {} \n config is {}'.format(i, config_id, config))
            while config_id in self.searched_params:
                config, config_id = self.generate_config()
            self.searched_params.append(config_id)
            config['note'] = config_id
            config['gpu'] = self.gm.wait_free_gpu()

            # run n times, model with large parameters may run just one time cause the card memory
            for i_task in range(self.args.n_run_few_epoch):
                config['seed'] = self.seeds[i_task]
                cmd = config2cmd(config)
                p = subprocess.Popen(cmd, shell=True)
                proc.append(p)
                time.sleep(1)
            if args.test is not True: time.sleep(15)
            print(config_id, i)
        for p in proc: p.wait()  # wait for all proc down
        self.log('Search complete !', with_time=True)

    def generate_config(self):
        config = {
            'dataset': self.args.dataset,
            'dataset_root': self.args.dataset_root,
            'seed': self.args.seed,
            'split_seed': self.args.split_seed,
            'hid_dim_alpha': choice([1, 2, 3, 4, 6]),
            'e_dim': choice([256, 512, 1024, 2048]),  # Excitation

            'mol_block': choice(['_TripletMessage', '_NNConv', '_TripletMessageLight', '_GCNConv', '_GATConv']),
            'message_steps': choice([1, 2, 3, 6]),
            'mol_readout': choice(['Set2Set', 'GlobalPool5', 'GlobalLAPool', ]),

            'pre_do': choice(['_None()', '_None()', 'Dropout(0.1)']),
            'graph_do': choice(['_None()', '_None()', 'Dropout(0.1)']),
            'flat_do': choice(['_None()', 'Dropout(0.1)', 'Dropout(0.2)', 'Dropout(0.5)']),
            'end_do': choice(['_None()', 'Dropout(0.1)', 'Dropout(0.2)', 'Dropout(0.5)']),

            'pre_norm': choice(['_None', '_BatchNorm', '_LayerNorm']),  # '_None'
            'graph_norm': choice(['_None', '_None', '_None', '_BatchNorm', '_LayerNorm', '_PairNorm']),
            'flat_norm': choice(['_None', '_None', '_None', '_BatchNorm', '_LayerNorm']),
            'end_norm': choice(['_None', '_None', '_None', '_BatchNorm', '_LayerNorm']),

            'pre_act': choice(['_None', 'ReLU', 'LeakyReLU', 'RReLU', 'RReLU', 'RReLU']),
            'graph_act': choice(['_None', 'ReLU', 'LeakyReLU', 'RReLU', 'RReLU', 'RReLU', 'CELU']),
            'flat_act': choice(['_None', 'ReLU', 'LeakyReLU', 'RReLU', 'RReLU', 'RReLU', 'CELU']),
            # 'end_act': choice(['_None', 'ReLU', 'LeakyReLU', 'RReLU', 'RReLU', 'RReLU', 'CELU']),
            'graph_res': choice([1, 0]),

            'loss': choice(['bcel']),
            'batch_size': choice([4, 8, 12, 16, 32, 64, 128, 256, 512, 768]),
            'optim': choice(['Adam', 'Ranger']),  # 'SGD'
            'k': choice([1, 3, 6]),
            'epochs': 30,  # 30
            'lr': choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
            'early_stop_patience': 50,
        }
        if self.args.test is True:
            config['test'] = 1
        if config['optim'] != 'Ranger':
            del config['k']
        if self.args.dataset in ['demo', 'bbbp', 'bace', 'sider', 'toxcast', 'tox21']:
            config['loss'] = choice(['bcel'])
        elif self.args.dataset in ['esol', 'freesolv', 'lipophilicity', 'physprop_perturbed']:
            config['loss'] = choice(['mse', 'mse', 'mse', 'mae', 'huber'])
        config_id = md5(' '.join([k + ' ' + str(v) for k, v in config.items()]))
        return config, config_id

    def auto_blend(self):
        self.log('Run more epochs estimation...')
        self.inferencer.evaluate_top_configs(top_n=self.args.n_top_blend, n_seed=self.args.n_run_full_epoch)
        self.log('Run solution for original test set...')
        self.inferencer.blend_and_inference()
        if args.dataset in ['physprop_perturbed']:
            self.inferencer.perturb_and_inference()

    def log(self, msg=None, with_time=False):
        msg = str(msg)
        if with_time: msg = msg + ' time elapsed {:.2f} hrs ({:.1f} mins)'.format(
            (time.time() - self.start) / 3600.,
            (time.time() - self.start) / 60.
        )
        with open(self.logs_dir / 'solver_log.txt', 'a+') as f:
            f.write(msg + '\n')
            print(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='physprop_perturbed', help='bindingdb_c,  lit_esr1ant')
    parser.add_argument('--dataset_root', type=str, default='../../Dataset/GLAM-GP', help='./demo')
    parser.add_argument('--n_init_configs', default=100, type=int, help='n initialized configurations')
    parser.add_argument('--n_run_few_epoch', default=3, type=int, help='3 run for a configuration')
    parser.add_argument('--n_top_blend', default=3, type=int, help='auto blend n models')
    parser.add_argument('--n_run_full_epoch', default=5, type=int, help='n run for full epochs with a config')
    parser.add_argument('--seed', default=1234, type=int, help='seed to init a model')
    parser.add_argument('--split_seed', default=1234, type=int, help='seed to split a dataset')
    parser.add_argument('--test', default=0, type=bool)

    args = parser.parse_args()
    solver = GLAM(args)
    solver.random_search()
    solver.auto_blend()
