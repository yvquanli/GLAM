import argparse
from pathlib import Path
import time
import subprocess
from random import choice
from utils import GPUManager, config2cmd, md5
from trainer import GLAMHelper
from dataset import Dataset


class GLAM():
    def __init__(self, args):
        self.args = args
        self.gm = GPUManager()
        self.seeds = [1, 12, 123, 1234, 16, 32, 64, 128]
        self.start = time.time()
        self.logs_dir = Path('./log_{}/'.format(args.dataset))
        Path.mkdir(self.logs_dir, exist_ok=True)
        self.glam_helper = GLAMHelper(args.dataset)
        self.searched_params = []

        self.log(msg=args.__dict__)
        self.log('Solver for {} running start @ {}'.format(args.dataset, time.asctime(time.localtime(self.start))))
        # assert len(self.gm.gpus) > 0
        self.log('{} gpus available'.format(len(self.gm.gpus)))

    def low_fidelity_training(self):
        proc = []
        for i in range(self.args.n_init_configs):
            config, config_id = self.sample_config()  # # take a point out of the search space
            self.log('Configuration {} start: \n config_id is {} \n config is {}'.format(i, config_id, config))
            while config_id in self.searched_params:
                config, config_id = self.sample_config()
            self.searched_params.append(config_id)
            config['note'] = config_id
            config['gpu'] = self.gm.wait_free_gpu(0.6)

            # run n times, model with large parameters may run just one time cause the card memory
            for i_task in range(self.args.n_low_fidelity_seed):
                config['seed'] = self.seeds[i_task]
                cmd = config2cmd(config)
                p = subprocess.Popen(cmd, shell=True)
                proc.append(p)
                time.sleep(10)
            time.sleep(60)
            print(config_id, i)
        for p in proc: p.wait()  # wait for all proc down
        self.log('Search complete !', with_time=True)

    def sample_config(self):
        config = {
            'dataset': self.args.dataset,  # 'bindingdb_c', 'lit_pkm2', 'lit_cat2a'
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
            'end_act': choice(['_None', 'ReLU', 'LeakyReLU', 'RReLU', 'RReLU', 'RReLU', 'CELU']),
            'graph_res': choice([1, 0]),

            'loss': choice(['bcel']), # bce
            'batch_size': choice([4, 8, 12, 16, 32, 64, 128, 256, 512, 768]), 
            'optim': choice(['Adam', 'Ranger']),  # 'SGD'
            'k': choice([1, 3, 6]),
            'epochs': 20,  # 20,
            'lr': choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
            'early_stop_patience': 50,
        }
        if config['optim'] == 'Adam':
            del config['k']
        config_id = md5(' '.join([k + ' ' + str(v) for k, v in config.items()]))
        return config, config_id

    def auto_blend(self):
        self.glam_helper.high_fidelity_training(top_n=self.args.n_top_blend, n_seed=self.args.n_high_fidelity_seed)
        self.glam_helper.blend_and_inference()

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
    parser.add_argument('--dataset', type=str, default='drugbank_caster', help='drugbank_caster')
    parser.add_argument('--n_init_configs', default=50, type=int, help='n initialized configurations')
    parser.add_argument('--n_low_fidelity_seed', default=1, type=int, help='3 run for a configuration')
    parser.add_argument('--n_top_blend', default=3, type=int, help='auto blend n models')
    parser.add_argument('--n_high_fidelity_seed', default=2, type=int, help='n run for full epochs with a config')
    parser.add_argument('--seed', default=1234, type=int, help='seed to init a model')
    args = parser.parse_args()
    solver = GLAM(args)
    solver.low_fidelity_training()
    solver.auto_blend()
