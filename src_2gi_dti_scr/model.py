import numpy as np
import torch
from torch_geometric.nn import Set2Set
from layer import _TripletMessage, _TripletMessageLight, _NNConv, _GCNConv, _GATConv
from layer import _None
from layer import GlobalPool5, GlobalLAPool, dot_and_global_pool2
from layer import LinearBlock, MessageBlock

from torch import nn
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap, \
    global_mean_pool as gep, global_sort_pool


class Architecture(torch.nn.Module):  # back to src v6.7
    def __init__(self, mol_in_dim=15, pro_in_dim=49,
                 mol_edge_in_dim=4, pro_edge_in_dim=8,
                 hid_dim_alpha=4, e_dim=1024, out_dim=1,
                 mol_block='_NNConv', pro_block='_GCNConv', message_steps=3,
                 mol_readout='GlobalPool5', pro_readout='GlobalPool5',
                 pre_norm='_None', graph_norm='_None', flat_norm='_None', end_norm='_None',
                 pre_do='_None()', graph_do='Dropout(0.2)', flat_do='_None()', end_do='Dropout(0.2)',
                 pre_act='RReLU', graph_act='RReLU', flat_act='RReLU', end_act='RReLU',
                 graph_res=True,
                 ):
        super(Architecture, self).__init__()
        hid_dim = mol_in_dim * hid_dim_alpha
        self.mol_lin0 = LinearBlock(mol_in_dim, hid_dim, norm=pre_norm, dropout=pre_do, act=pre_act)
        self.pro_lin0 = LinearBlock(pro_in_dim, hid_dim, norm=pre_norm, dropout=pre_do, act=pre_act)
        self.mol_conv = MessageBlock(hid_dim, hid_dim, mol_edge_in_dim,
                                     norm=graph_norm, dropout=graph_do, conv=mol_block, act=graph_act, res=graph_res)
        self.pro_conv = MessageBlock(hid_dim, hid_dim, pro_edge_in_dim,
                                     norm=graph_norm, dropout=graph_do, conv=pro_block, act=graph_act, res=graph_res)
        self.message_steps = message_steps

        exec('self.mol_readout = {}(in_channels=hid_dim, processing_steps=3)'.format(mol_readout))
        exec('self.pro_readout = {}(in_channels=hid_dim, processing_steps=3)'.format(pro_readout))
        _mol_ro = 5 if mol_readout == 'GlobalPool5' else 2
        _pro_ro = 5 if pro_readout == 'GlobalPool5' else 2
        self.mol_flat = LinearBlock(_mol_ro * hid_dim, hid_dim, norm=flat_norm, dropout=flat_do, act=flat_act)
        self.pro_flat = LinearBlock(_pro_ro * hid_dim, hid_dim, norm=flat_norm, dropout=flat_do, act=flat_act)

        self.lin_out0 = LinearBlock(hid_dim * 2 + message_steps * 2, e_dim, norm=end_norm, dropout=end_do, act=end_act)
        self.lin_out1 = LinearBlock(e_dim, out_dim, norm=end_norm, dropout=end_do, act='_None')

    def forward(self, data_mol, data_pro):
        # pre linear
        xm = self.mol_lin0(data_mol.x, batch=data_mol.batch)
        xp = self.pro_lin0(data_pro.x, batch=data_pro.batch)

        # graph conv
        hm, hp = None, None
        fusion = []
        for i in range(self.message_steps):
            xm, hm = self.mol_conv(xm, data_mol.edge_index, data_mol.edge_attr, h=hm, batch=data_mol.batch)
            xp, hp = self.pro_conv(xp, data_pro.edge_index, data_pro.edge_attr, h=hp, batch=data_pro.batch)
            fusion.append(dot_and_global_pool2(xm, xp, data_mol.batch, data_pro.batch))

        # readout
        outm = self.mol_readout(xm, data_mol.batch)
        outm = self.mol_flat(outm)
        outp = self.pro_readout(xp, data_pro.batch)
        outp = self.pro_flat(outp)

        # final linear
        out = torch.cat([outm, outp, torch.cat(fusion, dim=-1)], dim=-1)
        out = self.lin_out0(out)
        out = self.lin_out1(out)
        return out


Model = Architecture

if __name__ == '__main__':
    mol_batch, pro_batch = torch.load('./other/test_batch.pt')
    model = Model()
    result = model(mol_batch, pro_batch)
    a = 3
    # train, val, test, mol_datas, pro_datas = load_dataset_bindingdb('../dataset/')
