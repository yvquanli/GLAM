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
from utils import screening_metrics, regression_metrics, binary_metrics, get_loss
from utils import auto_metrics, read_logs, auto_dataset, model_args
from dataset import extract_batch_data
from utils import GPUManager, auto_summarize_logs, config2cmd, seed_torch

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

    def write_datasets(self):
        self.log('Writing datasets...')
        dataloaders = {
            'train': self.train_dataloader,
            'valid': self.valid_dataloader,
            'test': self.test_dataloader
        }
        save_dir = Path(self.train_dataloader.dataset.root) / 'split' / self.args.dataset
        Path.mkdir(save_dir.parent, exist_ok=True)
        Path.mkdir(save_dir, exist_ok=True)
        for name, dataloader in dataloaders.items():
            with open(save_dir / '{}.txt'.format(name), 'a+') as f:
                for i, id_batch in enumerate(dataloader):
                    for smi, pro, y in zip(id_batch.smi, id_batch.pro, id_batch.y):
                        f.write('{} {} {}\n'.format(smi, pro, y))

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


class TrainerRegression(Trainer):
    def __init__(self, args, model, train_dataset, valid_dataset, test_dataset=None, print_log=True):
        super(TrainerRegression, self).__init__(args, model, train_dataset, valid_dataset, test_dataset, print_log)
        self.metrics_fn = regression_metrics

    def train_iterations(self):
        self.model.train()
        losses = []
        for i, id_batch in enumerate(self.train_dataloader):
            if id_batch.__sizeof__() <= 1: continue   # bn will raise error if there is only one GLAM
            self.optimizer.zero_grad()
            mol_batch, pro_batch = extract_batch_data(self.train_dataloader.dataset, id_batch)
            mol_batch = mol_batch.to(self.device)
            pro_batch = pro_batch.to(self.device)
            y = id_batch.y.float().to(self.device).view(-1, 1)
            output = self.model(mol_batch, pro_batch).view(-1, 1)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if i % self.args.verbose_patience == 0:
                self.log('\tbatch {} training loss: {:.5f}'.format(i, loss.item()), with_time=True)
        trn_loss = torch.tensor(losses).mean()
        return trn_loss

    @torch.no_grad()
    def valid_iterations(self, mode='valid'):
        self.model.eval()
        dataloader = self.valid_dataloader if mode == 'valid' else self.test_dataloader
        losses, labels, pred_labels = [], [], []
        for id_batch in dataloader:
            if id_batch.__sizeof__() <= 1: continue   # bn will raise error if there is only one GLAM
            mol_batch, pro_batch = extract_batch_data(self.train_dataloader.dataset, id_batch)
            mol_batch = mol_batch.to(self.device)
            pro_batch = pro_batch.to(self.device)
            y = id_batch.y.float().to(self.device)
            output = self.model(mol_batch, pro_batch).view(-1)
            loss = self.criterion(output, y)
            losses.append(loss.cpu())
            labels.append(y.cpu())
            pred_labels.append(output.cpu())
        mean_loss = torch.tensor(losses).mean()
        if mode == 'inference': return torch.cat(labels), torch.cat(pred_labels)
        result = self.metrics_fn(torch.cat(labels).numpy(), torch.cat(pred_labels).numpy())
        return mean_loss, result


class TrainerBinaryClassification(Trainer):
    def __init__(self, args, model, train_dataset, valid_dataset, test_dataset=None, print_log=True):
        super(TrainerBinaryClassification, self).__init__(args, model, train_dataset, valid_dataset, test_dataset,
                                                          print_log)
        self.metrics_fn = binary_metrics

    def train_iterations(self):
        self.model.train()
        losses = []
        for i, id_batch in enumerate(self.train_dataloader):
            if id_batch.__sizeof__() <= 1: continue   # bn will raise error if there is only one GLAM
            self.optimizer.zero_grad()
            mol_batch, pro_batch = extract_batch_data(self.train_dataloader.dataset, id_batch)
            mol_batch = mol_batch.to(self.device)
            pro_batch = pro_batch.to(self.device)
            y = id_batch.y.to(self.device)
            output = self.model(mol_batch, pro_batch)
            loss = self.criterion(output, y)
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
        losses, labels, pred_scores, pred_labels = [], [], [], []
        for id_batch in dataloader:
            if id_batch.__sizeof__() <= 1: continue   # bn will raise error if there is only one GLAM
            mol_batch, pro_batch = extract_batch_data(self.train_dataloader.dataset, id_batch)
            mol_batch = mol_batch.to(self.device)
            pro_batch = pro_batch.to(self.device)
            y = id_batch.y.to(self.device)
            output = self.model(mol_batch, pro_batch)
            loss = self.criterion(output, y)
            losses.append(loss.item())
            output = torch.softmax(output, 1)  # for classification
            pred_score = output[:, 1]
            pred_label = torch.argmax(output, dim=1)
            pred_scores.append(pred_score.cpu())
            pred_labels.append(pred_label.cpu())
            labels.append(y.cpu())
        mean_loss = torch.tensor(losses).mean()
        if mode == 'inference': return torch.cat(labels), torch.cat(pred_labels), torch.cat(pred_scores)
        result = self.metrics_fn(torch.cat(labels).numpy(), y_score=torch.cat(pred_scores).numpy(),
                                 y_pred=torch.cat(pred_labels).numpy())
        return mean_loss, result


class TrainerScreening(TrainerBinaryClassification):
    def __init__(self, args, model, train_dataset, valid_dataset, test_dataset=None, print_log=True):
        super(TrainerScreening, self).__init__(args, model, train_dataset, valid_dataset, test_dataset, print_log)
        self.metrics_fn = screening_metrics
        if args.loss == 'wce':
            self.criterion = torch.nn.CrossEntropyLoss(weight=train_dataset.weight.to(self.device))


class TrainerBinary(Trainer):
    def __init__(self, args, model, train_dataset, valid_dataset, test_dataset=None):
        super(TrainerBinary, self).__init__(args, model, train_dataset, valid_dataset, test_dataset)

    def train_iterations(self):
        self.model.train()
        losses = []
        for i, id_batch in enumerate(self.train_dataloader):
            if id_batch.__sizeof__() <= 1: continue   # bn will raise error if there is only one GLAM
            self.optimizer.zero_grad()
            mol_batch, pro_batch = extract_batch_data(self.train_dataloader.dataset, id_batch)
            mol_batch = mol_batch.to(self.device)
            pro_batch = pro_batch.to(self.device)
            id_batch.y = id_batch.y.float().to(self.device)
            output = self.model(mol_batch, pro_batch).view(-1)
            output = torch.sigmoid(output)  # for binary
            loss = self.criterion(output, id_batch.y)  # bce loss
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
        losses = []
        labels = np.array([])
        outputs = np.array([])
        for id_batch in dataloader:
            if id_batch.__sizeof__() <= 1: continue   # bn will raise error if there is only one GLAM
            self.optimizer.zero_grad()
            mol_batch, pro_batch = extract_batch_data(self.train_dataloader.dataset, id_batch)
            mol_batch = mol_batch.to(self.device)
            pro_batch = pro_batch.to(self.device)
            y = id_batch.y.float().to(self.device)
            output = self.model(mol_batch, pro_batch).view(-1)
            output = torch.sigmoid(output)  # for binary

            # val metrics
            loss = self.criterion(output, y)
            losses.append(loss.item())
            labels = np.concatenate([labels, y.cpu().numpy()])
            outputs = np.concatenate([outputs, output.cpu().numpy()])
        mean_loss = np.array(losses).mean()
        result = binary_metrics(labels, outputs)
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
            seed_torch(seed=1234)
            args, dataset, _Trainer = auto_dataset(args)
            model = Model(dataset.mol_num_node_features, dataset.pro_num_node_features,
                          dataset.mol_num_edge_features, dataset.pro_num_edge_features, **model_args(args))
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
        if args.dataset in ['ADRB2', 'ALDH1', 'ESR1_ago', 'ESR1_ant', 'FEN1', 'GBA', 'IDH1', 'KAT2A', 'MAPK1',
                              'MTORC1', 'OPRK1', 'PKM2', 'PPARG', 'TP53', 'VDR']:
            self.log(self.blend_binary_classification(outputs, metrics_fn=screening_metrics))
        elif args.dataset in ['bindingdb_c']:
            self.log(self.blend_binary_classification(outputs, metrics_fn=binary_metrics))
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
        # logs_pd_selected = logs_pd[index].sort_values(metrics[0], ascending=False).iloc[:n_blend, :]
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
            config['epochs'] = 2000
            config['note'] = 'more_epochs_run'
            for seed in [1, 12, 123]:
                config['seed'] = seed
                config['gpu'] = gm.wait_free_gpu(thre=0.5)
                cmd = config2cmd(config)
                p = subprocess.Popen(cmd, shell=True)
                procs.append(p)
                time.sleep(60)
        for p in procs:
            p.wait()
        self.log('Run Complete!', with_time=True)

    @staticmethod
    def blend_binary_classification(outputs: list, opt='vote', metrics_fn=binary_metrics):
        ls, pls, ss = [], [], []
        for _l, _pl, _s in outputs:
            ls.append(_l)
            pls.append(_pl)
            ss.append(_s)
        blendd_l = ls[0]
        blendd_pl = torch.stack(pls, dim=1).mode(dim=1)[0] if opt == 'vote' else None
        blendd_ss = torch.stack(ss, dim=1).mean(dim=1)
        return metrics_fn(blendd_l.numpy(), y_score=blendd_ss.numpy(), y_pred=blendd_pl.numpy())

    def log(self, msg=None, with_time=False):
        msg = str(msg)
        if with_time: msg = msg + ' time elapsed {:.2f} hrs ({:.1f} mins)'.format(
            (time.time() - self.start) / 3600.,
            (time.time() - self.start) / 60.
        )
        with open(self.logs_dir / 'inference_log.txt', 'a+') as f:
            f.write(msg + '\n')
            print(msg)


if __name__ == '__main__':
    inf = GLAMHelper('ESR1_ant', n_blend=3)
    inf.blend_and_inference()

    datasets = ['ADRB2', 'ALDH1', 'ESR1_ago', 'ESR1_ant', 'FEN1', 'GBA', 'IDH1',
                'KAT2A', 'MAPK1', 'MTORC1', 'OPRK1', 'PPARG', 'TP53']
    for dataset in datasets:
        # results = auto_summarize_logs(dataset, ongoing=True)
        inf = GLAMHelper(dataset, n_blend=3)
        inf.blend_and_inference()

    # trainer = Trainer(None, None, None, None)
