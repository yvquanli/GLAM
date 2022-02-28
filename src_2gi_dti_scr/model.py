import torch
import numpy as np
from torch_geometric.nn import Set2Set
from layer import _None
from layer import GlobalPool5, GlobalLAPool, dot_and_global_pool2
from layer import LinearBlock, MessageBlock


class Architecture(torch.nn.Module):  # back to src v6.7
    def __init__(self, mol_in_dim=15,
                 mol_edge_in_dim=4,
                 hid_dim_alpha=4, e_dim=1024, out_dim=1,
                 mol_block='_NNConv', message_steps=3,
                 mol_readout='GlobalPool5',
                 pre_norm='_None', graph_norm='_None', flat_norm='_None', end_norm='_None',
                 pre_do='_None()', graph_do='Dropout(0.2)', flat_do='_None()', end_do='Dropout(0.2)',
                 pre_act='RReLU', graph_act='RReLU', flat_act='RReLU', end_act='RReLU',
                 graph_res=True,
                 ):
        super(Architecture, self).__init__()
        hid_dim = mol_in_dim * hid_dim_alpha
        self.mol1_lin0 = LinearBlock(mol_in_dim, hid_dim, norm=pre_norm, dropout=pre_do, act=pre_act)
        self.mol2_lin0 = LinearBlock(mol_in_dim, hid_dim, norm=pre_norm, dropout=pre_do, act=pre_act)
        self.mol1_conv = MessageBlock(hid_dim, hid_dim, mol_edge_in_dim,
                                      norm=graph_norm, dropout=graph_do, conv=mol_block, act=graph_act, res=graph_res)
        self.mol2_conv = MessageBlock(hid_dim, hid_dim, mol_edge_in_dim,
                                      norm=graph_norm, dropout=graph_do, conv=mol_block, act=graph_act, res=graph_res)
        self.message_steps = message_steps

        exec('self.mol1_readout = {}(in_channels=hid_dim, processing_steps=3)'.format(mol_readout))
        exec('self.mol2_readout = {}(in_channels=hid_dim, processing_steps=3)'.format(mol_readout))
        _mol_ro = 5 if mol_readout == 'GlobalPool5' else 2
        self.mol1_flat = LinearBlock(_mol_ro * hid_dim, hid_dim, norm=flat_norm, dropout=flat_do, act=flat_act)
        self.mol2_flat = LinearBlock(_mol_ro * hid_dim, hid_dim, norm=flat_norm, dropout=flat_do, act=flat_act)

        self.lin_out0 = LinearBlock(hid_dim * 2 + message_steps * 2, e_dim, norm=end_norm, dropout=end_do, act=end_act)
        self.lin_out1 = LinearBlock(e_dim, out_dim, norm=end_norm, dropout=end_do, act='_None')

    def forward(self, mol1, mol2):
        # pre linear
        xm1 = self.mol1_lin0(mol1.x, batch=mol1.batch)
        xm2 = self.mol2_lin0(mol2.x, batch=mol2.batch)

        # graph conv
        hm1, hm2 = None, None
        fusion = []
        for i in range(self.message_steps):
            xm1, hm1 = self.mol1_conv(xm1, mol1.edge_index, mol1.edge_attr, h=hm1, batch=mol1.batch)
            xm2, hm2 = self.mol2_conv(xm2, mol2.edge_index, mol2.edge_attr, h=hm2, batch=mol2.batch)
            fusion.append(dot_and_global_pool2(xm1, xm2, mol1.batch, mol2.batch))

        # readout
        outm1 = self.mol1_readout(xm1, mol1.batch)
        outm2 = self.mol2_readout(xm2, mol2.batch)
        outm1 = self.mol1_flat(outm1)
        outm2 = self.mol2_flat(outm2)

        # final linear
        out = self.lin_out0(torch.cat([outm1, outm2, torch.cat(fusion, dim=-1)], dim=-1))
        out = self.lin_out1(out)
        return out


Model = Architecture

if __name__ == '__main__':
    mol_batch = torch.load('./other/test_batch.pt')
    model = Model()
    result = model(mol_batch)
    a = 3
