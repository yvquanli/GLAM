import torch
from torch_geometric.nn import Set2Set
from layer import _None
from layer import GlobalPool5, GlobalLAPool
from layer import LinearBlock, MessageBlock

def model_args(args):
    _other_args_name = ['dataset_root', 'dataset', 'split', 'seed', 'gpu', 'note', 'batch_size', 'epochs', 'loss',
                        'optim', 'k', 'lr', 'lr_reduce_rate', 'lr_reduce_patience', 'early_stop_patience',
                        'verbose_patience', 'split_seed', 'test']
    model_args_dict = {}
    for k, v in args.__dict__.items():
        if k not in _other_args_name:
            model_args_dict[k] = v
    return model_args_dict

def init_weith_with_gain(modules):
    for m in modules:
        if isinstance(m, LinearBlock):
            torch.nn.init.xavier_uniform_(m.linear.weight, gain=4)  # for pasp expr: avoid collapsed output of a predictor without training


class Architecture(torch.nn.Module):
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
        super(Architecture, self).__init__()
        hid_dim = mol_in_dim * hid_dim_alpha
        self.mol_lin0 = LinearBlock(mol_in_dim, hid_dim, norm=pre_norm, dropout=pre_do, act=pre_act)
        self.mol_conv = MessageBlock(hid_dim, hid_dim, mol_edge_in_dim,
                                     norm=graph_norm, dropout=graph_do, conv=mol_block, act=graph_act, res=graph_res)
        self.message_steps = message_steps

        exec('self.mol_readout = {}(in_channels=hid_dim, processing_steps=3)'.format(mol_readout))
        _mol_ro = 5 if mol_readout == 'GlobalPool5' else 2
        self.mol_flat = LinearBlock(_mol_ro * hid_dim, e_dim, norm=flat_norm, dropout=flat_do, act=flat_act)
        self.lin_out1 = LinearBlock(e_dim, out_dim, norm=end_norm, dropout=end_do, act='_None')
        # init_weith_with_gain(self.modules())  # for pasp expr

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
        return out


Model = Architecture

if __name__ == '__main__':
    mol_batch = torch.load('./other/test_batch.pt')
    model = Model()
    result = model(mol_batch)
