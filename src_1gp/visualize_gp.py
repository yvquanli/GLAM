import torch
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import Set2Set
import pathlib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib
from matplotlib import pyplot as plt
from utils import config2cmd, auto_dataset, model_args
from argparse import Namespace
from dataset import PertubationDataset

from layer import LinearBlock, MessageBlock


class GLAM(torch.nn.Module):
    def __init__(self, mol_in_dim=15,
                 mol_edge_in_dim=4,
                 hid_dim_alpha=4, e_dim=1024, out_dim=1,
                 mol_block='_NNConv', message_steps=3,
                 mol_readout='GlobalPool5',
                 pre_norm='_None', graph_norm='_None', flat_norm='_None', end_norm='_None',
                 pre_do='_None()', graph_do='Dropout(0.2)', flat_do='_None()', end_do='Dropout(0.2)',
                 pre_act='RReLU', graph_act='RReLU', flat_act='RReLU',  # end_act='RReLU',
                 graph_res=True,
                 ):
        super(GLAM, self).__init__()
        hid_dim = mol_in_dim * hid_dim_alpha
        self.mol_lin0 = LinearBlock(mol_in_dim, hid_dim, norm=pre_norm, dropout=pre_do, act=pre_act)
        self.mol_conv = MessageBlock(hid_dim, hid_dim, mol_edge_in_dim,
                                     norm=graph_norm, dropout=graph_do, conv=mol_block, act=graph_act, res=graph_res)
        self.message_steps = message_steps

        exec('self.mol_readout = {}(in_channels=hid_dim, processing_steps=3)'.format(mol_readout))
        _mol_ro = 5 if mol_readout == 'GlobalPool5' else 2
        self.mol_flat = LinearBlock(_mol_ro * hid_dim, e_dim, norm=flat_norm, dropout=flat_do, act=flat_act)
        self.lin_out1 = LinearBlock(e_dim, out_dim, norm=end_norm, dropout=end_do, act='_None')

    def forward(self, data_mol):
        # pre linear
        xm = self.mol_lin0(data_mol.x, batch=data_mol.batch)

        # graph conv
        hm = None
        for i in range(self.message_steps):
            xm, hm = self.mol_conv(xm, data_mol.edge_index, data_mol.edge_attr, h=hm, batch=data_mol.batch)

        # readout
        outm = self.mol_readout(xm, data_mol.batch)

        # final linear
        outm = self.mol_flat(outm)
        out = self.lin_out1(outm)
        return out, xm


Model = GLAM


class Visualizer:
    def __init__(self, ckpt_root, save_root='./out_imgs', vis_content='hidden_node'):
        self.ckpt_root = Path(ckpt_root)
        self.save_root = Path(save_root)
        self.vis_content = vis_content
        self.colormap = 'RdBu'
        self.device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        self.args, self.model, self.dataset = self.load_model()
        Path.mkdir(self.save_root, exist_ok=True)

    def load_model(self):
        ckpt = torch.load(str(self.ckpt_root / 'best_save.ckpt'), map_location=self.device)
        args = ckpt['args']
        # assert args.mol_readout == 'Set2Set'  # or GlobalLAPool
        # assert there is no Batch norm
        args, dataset, _Trainer = auto_dataset(args)
        model = Model(dataset.mol_num_node_features, dataset.mol_num_edge_features, **model_args(args))
        model.load_state_dict(ckpt['model_state_dict'])
        return args, model, dataset

    def visualize(self):
        dataloader = DataLoader(self.dataset.test, batch_size=1)
        for data in dataloader:
            if self.vis_content == 'set2set_attention':
                weights, y_pred, y_true = self.set2set_attention_to_weight(data)
            elif self.vis_content == 'lapool_attention':
                weights, y_pred, y_true = self.lapool_attention_to_weight(data)
            elif self.vis_content == 'hidden_node':
                weights, y_pred, y_true = self.hidden_node_to_weight(data)
            else:
                raise ValueError('Unknown content to visualize')
            # weights = self.stand_scale(weights)
            smiles = data.smi[0]
            fig = self.visualize_atom_attention(smiles, weights, self.colormap)
            save_path = self.save_root / '{}_pred{:.3f}_true{:.3f}_{}.png'.format(self.vis_content, y_pred, y_true,
                                                                                  smiles)
            self.save_fig(fig, save_path)
            # break

    def save_fig(self, fig, save_path):
        fig.savefig(save_path, bbox_inches='tight', dpi=400, pad_inches=0)
        plt.close(fig)
        print('Saved!', save_path)

    def hidden_node_to_weight(self, data):
        # get attention
        mol_batch = data.to(self.device)
        y_pred, xm = self.model(mol_batch)
        weight = xm.mean(dim=-1).detach().cpu().numpy()
        y_pred = y_pred.detach().item()
        y_true = data.y.to(self.device).view(-1).item()
        return weight, y_pred, y_true

    @staticmethod
    def norm(tensor, p=2):
        return torch.nn.functional.normalize(tensor, p=p)

    @staticmethod
    def stand_scale(X):
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, 1))
        # X_scaled[X_scaled < 0] = 0
        return X_scaled

    @staticmethod
    def visualize_atom_attention(smiles: str, weights, colormap: str):
        mol = Chem.MolFromSmiles(smiles)
        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, colorMap=plt.get_cmap(colormap), alpha=0,
                                                         size=(150, 150))
        return fig


if __name__ == '__main__':
    viser = Visualizer(ckpt_root='./other/ckpt4vis_set2set', vis_content='hidden_node')
    viser.visualize()
