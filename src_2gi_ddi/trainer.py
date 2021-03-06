import time
import torch
import shutil
import pathlib
import subprocess
import numpy as np
from argparse import Namespace
import pandas as pd
from datetime import datetime
from os.path import join
from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam, SGD, lr_scheduler
from ranger import Ranger
from torch_geometric.data import DataLoader
from model import Model
from utils import screening_metrics, regression_metrics, binary_metrics, multi_class_metrics, get_loss
from utils import auto_metrics, read_logs, auto_dataset, model_args
from utils import GPUManager, auto_summarize_logs, config2cmd
from dataset import extract_batch_data, Dataset

torch.backends.cudnn.enabled = True


class Trainer():
    def __init__(self, args, model, train_dataset, valid_dataset, test_dataset=None, print_log=True):
        self.args = args  # ;  print(args)
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Setting the train valid and test data loader
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=32)
        if test_dataset: self.test_dataloader = DataLoader(test_dataset, batch_size=32)

        # Setting the loss, optimizer, scheduler
        self.criterion = get_loss(args.loss)
        if args.optim == 'Ranger':
            self.optimizer = Ranger(self.model.parameters(), lr=args.lr, k=args.k)  # weight_decay=args.weight_decay
        elif args.optim == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        elif args.optim == 'SGD':
            self.optimizer = SGD(self.model.parameters(), lr=args.lr)
        else:
            raise Exception('Error optimizer argv')
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=args.lr_reduce_rate,
                                                        patience=args.lr_reduce_patience, min_lr=1e-6)

        # other
        self.start = time.time()
        self.print_log = print_log
        cur_abs_file_dir = Path(__file__).resolve().parent
        save_id = datetime.utcnow().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3] + '_' + 'seed_' + str(args.seed)
        self.log_save_dir = cur_abs_file_dir / ('log_' + self.args.dataset) / '{}'.format(save_id)
        Path.mkdir(self.log_save_dir.parent, exist_ok=True)
        Path.mkdir(self.log_save_dir, exist_ok=True)
        self.records = {'val_losses': []}

        self.log(msgs=['\t{}:{}\n'.format(k, v) for k, v in args.__dict__.items()])
        self.log('save id: {}'.format(save_id))
        self.log('run device: {}'.format(self.device))
        self.log('train set num:{}    valid set num:{}    test set num: {}'.format(
            len(train_dataset), len(valid_dataset), len(test_dataset)))
        self.log("total parameters:" + str(sum([p.nelement() for p in self.model.parameters()])))
        self.log(msgs=str(model).split('\n'))

    def train(self):
        self.log('Training start...')
        early_stop_cnt = 0
        for epoch in tqdm(range(self.args.epochs)):
            trn_loss = self.train_iterations()
            val_loss, result = self.valid_iterations()
            self.scheduler.step(val_loss)
            lr_cur = self.scheduler.optimizer.param_groups[0]['lr']
            self.log('Epoch:{} trn_loss:{:.5f} val_loss:{:.5f} val_result:{} lr_cur:{:.7f}'.format(
                epoch, trn_loss, val_loss, result, lr_cur), with_time=True)
            self.records['val_losses'].append(val_loss)
            if val_loss == np.array(self.records['val_losses']).min():
                self.save_ckpt(epoch)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if 0 < self.args.early_stop_patience < early_stop_cnt:
                self.log('Early stop hitted!')
                break
        self.save_ckpt(epoch, final_save=True)

    def train_and_test(self):
        self.train()
        self.log('Testing...')
        self.load_best_ckpt()
        val_loss, val_result = self.valid_iterations(mode='valid')
        test_loss, test_result = self.valid_iterations(mode='test')
        self.log(msg=str(vars(self.args)))
        loss_info = {'testloss': test_loss.item(), 'valloss': val_loss.item()}
        val_result_new = {}
        for k in val_result.keys():
            val_result_new['val' + k] = val_result[k]
        self.log('{}|{}|{}'.format(loss_info, test_result, val_result_new))

    def save_ckpt(self, epoch, final_save=False):
        file_name = 'final_save.ckpt' if final_save else 'best_save.ckpt'
        with open(join(self.log_save_dir, file_name), 'wb') as f:
            torch.save({
                'args': self.args,
                'records': self.records,
                'model_state_dict': self.model.state_dict(),
            }, f)
        self.log('Model saved at epoch {}'.format(epoch))

    def gen_test_batch(self):
        for batch in self.valid_dataloader:
            torch.save(batch, './other/test_batch.pt')
            break

    def load_best_ckpt(self):
        ckpt_path = join(self.log_save_dir, 'best_save.ckpt')
        self.log('The best ckpt is {}'.format(ckpt_path))
        self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path):
        self.log('Ckpt loading: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.args = ckpt['args']
        self.records = ckpt['records']
        self.model.load_state_dict(ckpt['model_state_dict'])

    def log(self, msg=None, msgs=None, with_time=False):
        if self.print_log is False: return
        if with_time: msg = msg + ' time elapsed {:.2f} hrs ({:.1f} mins)'.format(
            (time.time() - self.start) / 3600.,
            (time.time() - self.start) / 60.
        )
        with open(join(self.log_save_dir, 'log.txt'), 'a+') as f:
            if msgs:
                self.log('#' * 80)
                if '\n' not in msgs[0]: msgs = [m + '\n' for m in msgs]
                f.writelines(msgs)
                for x in msgs: print(x, end='')
                self.log('#' * 80)
            if msg:
                f.write(msg + '\n')
                print(msg)


class TrainerMolBinaryClassificationNANBCE(Trainer):
    def __init__(self, args, model, train_dataset, valid_dataset, test_dataset=None, print_log=True):
        super(TrainerMolBinaryClassificationNANBCE, self).__init__(args, model, train_dataset, valid_dataset,
                                                                    test_dataset, print_log)
        self.metrics_fn = binary_metrics

    def train_iterations(self):
        self.model.train()
        losses = []
        for i, data in enumerate(self.train_dataloader):
            if data.__sizeof__() <= 1: continue  # bn will raise error if there is only one GLAM
            mol_batch1, mol_batch2 = extract_batch_data(self.train_dataloader.dataset.mol_data, data)
            self.optimizer.zero_grad()
            mol_batch1, mol_batch2 = mol_batch1.to(self.device), mol_batch2.to(self.device)
            y_score = self.model(mol_batch1, mol_batch2)
            y_true = data.y.to(self.device)
            loss = self.criterion(y_score.view(-1), y_true.float())
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if i % self.args.verbose_patience == 0:
                self.log('\tbatch {} training loss: {:.5f}'.format(i, loss.item()), with_time=True)
        trn_loss = np.array(losses).mean()
        return trn_loss

    @torch.no_grad()
    def valid_iterations(self, mode='valid'):
        self.model.eval()
        dataloader = self.valid_dataloader if mode == 'valid' else self.test_dataloader
        losses, ys_true, ys_score, = [], [], []
        for data in dataloader:
            if data.__sizeof__() <= 1: continue  # bn will raise error if there is only one GLAM
            mol_batch1, mol_batch2 = extract_batch_data(self.train_dataloader.dataset.mol_data, data)
            self.optimizer.zero_grad()
            mol_batch1, mol_batch2 = mol_batch1.to(self.device), mol_batch2.to(self.device)
            y_score = self.model(mol_batch1, mol_batch2)
            y_true = data.y.to(self.device)
            loss = self.criterion(y_score.view(-1), y_true.float())
            losses.append(loss.item())
            ys_score.append(torch.sigmoid(y_score).cpu())  # add sigmoid
            ys_true.append(y_true.cpu())
        mean_loss = torch.tensor(losses).mean()
        ys_true, ys_score = torch.cat(ys_true, dim=0), torch.cat(ys_score, dim=0)
        if mode == 'inference': return ys_score, ys_true
        result = self.metrics_fn(ys_true.numpy(), ys_score.numpy())
        return mean_loss, result


class TrainerMolMultiClassificationNANBCE(Trainer):
    def __init__(self, args, model, train_dataset, valid_dataset, test_dataset=None, print_log=True):
        super(TrainerMolMultiClassificationNANBCE, self).__init__(args, model, train_dataset, valid_dataset,
                                                                  test_dataset, print_log)
        self.metrics_fn = multi_class_metrics

    def train_iterations(self):
        self.model.train()
        losses = []
        for i, data in enumerate(self.train_dataloader):
            if data.__sizeof__() <= 1: continue  # bn will raise error if there is only one GLAM
            mol_batch1, mol_batch2 = extract_batch_data(self.train_dataloader.dataset.mol_data, data)
            self.optimizer.zero_grad()
            mol_batch1, mol_batch2 = mol_batch1.to(self.device), mol_batch2.to(self.device)
            y_score = self.model(mol_batch1, mol_batch2)
            y_true = data.y.to(self.device)
            loss = self.criterion(y_score, y_true)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if i % self.args.verbose_patience == 0:
                self.log('\tbatch {} training loss: {:.5f}'.format(i, loss.item()), with_time=True)
        trn_loss = np.array(losses).mean()
        return trn_loss

    @torch.no_grad()
    def valid_iterations(self, mode='valid'):
        self.model.eval()
        dataloader = self.valid_dataloader if mode == 'valid' else self.test_dataloader
        losses, ys_true, ys_score, = [], [], []
        for data in dataloader:
            if data.__sizeof__() <= 1: continue  # bn will raise error if there is only one GLAM
            mol_batch1, mol_batch2 = extract_batch_data(self.train_dataloader.dataset.mol_data, data)
            self.optimizer.zero_grad()
            mol_batch1, mol_batch2 = mol_batch1.to(self.device), mol_batch2.to(self.device)
            y_score = self.model(mol_batch1, mol_batch2)
            y_true = data.y.to(self.device)
            loss = self.criterion(y_score, y_true)
            losses.append(loss.item())
            ys_score.append(torch.log_softmax(y_score, dim=1).cpu())  # add log_softmax
            ys_true.append(y_true.cpu())
        mean_loss = torch.tensor(losses).mean()
        ys_true, ys_score = torch.cat(ys_true, dim=0), torch.cat(ys_score, dim=0)
        if mode == 'inference': return ys_score, ys_true
        result = self.metrics_fn(ys_true.numpy(), ys_score.numpy())
        return mean_loss, result


class GLAMHelper():
    def __init__(self, dataset: str, n_blend=3):
        self.dataset = dataset
        self.n_blend = n_blend
        self.start = time.time()
        self.logs_dir = pathlib.Path('./log_{}/'.format(self.dataset))
        self.log('Inference process of {} start...'.format(dataset))

    def blend_and_inference(self, custom_dataset=None):
        self.log('Start to blend models and inference ...')
        ids, configs = self.select_top_config()
        outputs = []
        for id, config in zip(ids, configs):
            args = Namespace(**eval(config))
            args.gpu = 0  # all
            args, dataset, _Trainer = auto_dataset(args)
            model = Model(dataset.mol_num_node_features, dataset.mol_num_edge_features, **model_args(args))
            trainer = _Trainer(args, model, dataset.train, dataset.val, dataset.test, print_log=False)
            shutil.rmtree(trainer.log_save_dir)  # remove new made log directory
            if custom_dataset is not None:
                trainer.test_dataloader = DataLoader(custom_dataset, batch_size=32)
            trainer.log_save_dir = './log_{}/{}'.format(args.dataset, id)
            trainer.load_best_ckpt()
            self.log('Checkpoint {} loaded.'.format(id))
            output = trainer.valid_iterations(mode='inference')
            outputs.append(output)
            self.log('inference done!', with_time=True)
        self.log('blend results: ')
        if args.dataset in ['drugbank_caster']:
            self.log(self.blend_binary_classification_mt(outputs))
        else:
            raise ValueError('unknown dataset')
        self.log('Done!', with_time=True)

    def select_top_config(self):
        logs = read_logs(self.logs_dir)
        if len(logs) < 1:
            self.log('Error: There is no log files found in {}!'.format(self.logs_dir))
        logs_pd = pd.DataFrame(logs)
        # index = logs_pd.loc[:, 'epochs'] >= 100  # assert evaluation epochs > 100 > search epochs
        metrics = auto_metrics(self.dataset)
        n_blend = len(logs_pd) if len(logs_pd) < self.n_blend else self.n_blend
        self.log('{} checkpoints select!'.format(n_blend))
        logs_pd_selected = logs_pd.sort_values(metrics[0], ascending=False).iloc[:n_blend, :]
        self.log('More info about picked checkpoints can be found here: {}/inf_ckpt_selected.csv'.format(self.logs_dir))
        logs_pd_selected.to_csv(self.logs_dir / 'inf_ckpt_selected.csv')
        return logs_pd_selected['id'], logs_pd_selected['config']

    def high_fidelity_training(self, top_n, n_seed):
        self.log('Run configurations for more epochs to achieve better results...')
        logs_summary = auto_summarize_logs(self.dataset)
        gm = GPUManager()
        procs = []
        for i in range(top_n):
            config = logs_summary.iloc[i, :]['config']
            self.log('Configuration {}: {} ...'.format(i + 1, config))
            config = eval(config)
            config['epochs'] = 2  # 2000
            config['note'] = 'more_epochs_run'
            for seed in [1, 12, 123]:
                config['seed'] = seed
                config['gpu'] = gm.wait_free_gpu(thre=0.5)
                cmd = config2cmd(config)
                p = subprocess.Popen(cmd, shell=True)
                procs.append(p)
                time.sleep(30)
        for p in procs:
            p.wait()
        self.log('Run Complete!', with_time=True)

    @staticmethod
    def blend_regression(outputs: list, opt='mean'):
        ls, pls = [], []
        for _l, _pl in outputs:
            ls.append(_l)
            pls.append(_pl)
        blendd_l = ls[0]
        blendd_pl = torch.stack(pls, dim=1).mean(dim=1) if opt == 'mean' else None
        return regression_metrics(blendd_l.numpy(), y_pred=blendd_pl.numpy())

    @staticmethod
    def blend_binary_classification(outputs: list, opt='vote', metrics_fn=binary_metrics):
        ls, pls, ss = [], [], []
        for _l, _s in outputs:
            ls.append(_l)
            ss.append(_s)
        blendd_l = ls[0]
        blendd_ss = torch.stack(ss, dim=1).mean(dim=1)
        return metrics_fn(blendd_l.numpy(), y_score=blendd_ss.numpy())

    @staticmethod
    def blend_binary_classification_mt(outputs: list, opt='vote', metrics_fn=binary_metrics):
        ls, ss = [], []
        for _s, _l in outputs:
            ls.append(_l)
            ss.append(_s)
        blendd_l = ls[0]
        # blendd_pl = torch.stack(pls, dim=2).mode(dim=2)[0] if opt == 'vote' else None
        blendd_ss = torch.stack(ss, dim=2).mean(dim=2)
        return metrics_fn(blendd_l.numpy(), y_score=blendd_ss.numpy())

    @staticmethod
    def blend_multi_classification(outputs: list, opt='vote', metrics_fn=multi_class_metrics):
        ls, ss = [], []
        for _s, _l in outputs:
            ls.append(_l)
            ss.append(_s)
        blendd_l = ls[0]
        # blendd_pl = torch.stack(pls, dim=2).mode(dim=2)[0] if opt == 'vote' else None
        blendd_ss = torch.stack(ss, dim=2).mean(dim=2)
        return metrics_fn(blendd_l.numpy(), y_score=blendd_ss.numpy())

    def log(self, msg=None, with_time=False):
        msg = str(msg)
        if with_time: msg = msg + ' time elapsed {:.2f} hrs ({:.1f} mins)'.format(
            (time.time() - self.start) / 3600.,
            (time.time() - self.start) / 60.
        )
        with open(self.logs_dir / 'inference_log.txt', 'a+') as f:
            f.write(msg + '\n')
            print(msg)
