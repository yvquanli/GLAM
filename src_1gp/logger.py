import pathlib
import pandas as pd
from metrics import auto_metrics

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


def print_ongoing_info(logs_dir: pathlib.PosixPath):
    for p in logs_dir.glob('./*seed*'):
        with open(p / 'log.txt') as f:
            lines = f.readlines()
            if lines[-1].startswith('{'):
                continue
            lines.reverse()
            for i, line in enumerate(lines):
                if 'Model saved at epoch' in line:
                    print(p, '----------->', lines[i + 1], end='')
                    break


def auto_summarize_logs(dataset: str, ongoing=False):
    logs_dir = pathlib.Path('./log_{}/'.format(dataset))
    if not logs_dir.exists():
        return None
    print('\n\n', '#' * 30, dataset, '#' * 30)
    results = summarize_logs(logs_dir=logs_dir, metrics=auto_metrics(dataset))
    print_red('Ongoing task details')
    if ongoing:
        print_ongoing_info(logs_dir=logs_dir)
    return results


def config2cmd(config: dict):
    _ = ' '.join(['--' + k + ' ' + str(v) for k, v in config.items()])
    cmd = 'python3 run.py {}'.format(_)
    cmd = cmd.replace('(', '\(').replace(')', '\)')  # for shell run
    print(cmd)
    return cmd


def print_red(s: str):
    print("\033[1;31m{}\033[0m".format(s))


def read_logs(logs_dir: pathlib.PosixPath):
    logs = []
    for p in logs_dir.glob('./*seed*'):
        log_path = p / 'log.txt'
        with open(log_path) as f:
            # read a single logfile to lines, and skip the log files without test
            lines = f.readlines()
            if not lines[-1].startswith('{') :
                continue

            # read a sigle logfile to config from -2 line, and short the words for better visual experience
            str_config_dict = lines[-2].replace('\n', '').strip().replace('mol_', 'm').replace('pro_', 'p') \
                .replace('depth', 'd').replace('graph_res', 'res').replace('batch_size', 'bs') \
                .replace('_TripletMessage', 'Trim').replace('_NNConv', 'NN').replace('_GCNConv', 'GCN') \
                .replace('_GATConv', 'GAT').replace('hid_dim_alpha', 'a').replace('message_steps', 'step') \
                .replace('Dropout(', '(').replace('Global', '').replace('_norm', 'n') \
                .replace('_LayerNorm', 'LN').replace('_BatchNorm', 'BN').replace('_PairNorm', 'PN') \
                .replace('more_epochs_run', 'mer').replace('_None', '0') \
                .replace('LeakyReLU', 'LReLU')
            config_for_print = eval(str_config_dict)
            for item in ['dataset_root', 'seed', 'gpu', 'verbose_patience', 'out_dim',
                         'early_stop_patience', 'lr_reduce_rate', 'lr_reduce_patience']:
                del config_for_print[item]

            # read a single logfile to loss, test information, valid information.
            loss_info, test_info, valid_info = lines[-1].replace(
                '\n', '').strip().split('|')
            # print(p, loss_info, test_info)  # for some inf set
            log = {'id': p.name}
            if 'inf' in loss_info or 'inf' in test_info or 'inf' in valid_info: continue
            log.update(eval(loss_info))
            log.update(eval(test_info))
            log.update(eval(valid_info))
            log.update(config_for_print)
            log.update({'config': lines[-2]})
            logs.append(log)
    return logs


def summarize_logs(logs_dir: pathlib.PosixPath, metrics: list):
    logs = read_logs(logs_dir)
    if len(logs) >= 1:
        # group, sort, and print the logs
        logs_pd = pd.DataFrame(logs).sort_values(metrics[0], ascending=False)
        logs_summary = []
        for note, df in logs_pd.groupby('note'):
            d = {'id(note)': note, 'n_run': len(df), 'dataset': df['dataset'].iloc[0],
                 'config': df['config'].iloc[0]}
            for m in metrics:
                array = df[m].astype(float)
                for opt in ['mean', 'min', 'max', 'std']:
                    d[opt + m] = eval('array.{}()'.format(opt))
            d.update({})
            logs_summary.append(d)
        logs_summary = pd.DataFrame(logs_summary).sort_values(
            'mean' + metrics[0], ascending=False)
        save_path = str(logs_dir / 'logs_summary.csv')
        print_red(
            'Search Result Info, more info and config can be found in {}'.format(save_path))
        # print info without config
        print(logs_summary.drop(labels=['config'], axis=1))
        logs_summary.to_csv(save_path)

        # search results details in groups
        search_result = []
        groups = logs_pd.groupby('note')
        for note in logs_summary.loc[:, 'id(note)']:
            group = groups.get_group(note).sort_values(
                metrics[0], ascending=False)
            search_result.append(group)
        search_result = pd.concat(search_result)
        save_path = str(logs_dir / 'search_result.csv')
        print_red(
            'Detailed Search Result Info,more info and config can be found in {}'.format(save_path))
        # print info without config
        print(search_result.drop(labels=['config'], axis=1))
        search_result.to_csv(save_path)
        return logs_summary


if __name__ == "__main__":
    from dataset import dataset_names
    datasets = dataset_names['a']
    for dataset in datasets:
        results = auto_summarize_logs(dataset, ongoing=True)
