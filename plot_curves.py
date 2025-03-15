import argparse
import json
import os

import tikzplotlib
from matplotlib import pyplot as plt


class MetricAggregator:
    def __init__(self):
        self.metrics = []
        self.data = {}
        self.epochs = {}

    def clear(self):
        self.data.clear()
        self.metrics.clear()
        self.epochs.clear()

    def update(self, data: dict):
        for k, v in data.items():
            if k == 'epoch':
                continue
            if (metric := k[k.find('_') + 1:]) not in self.metrics:
                self.metrics.append(metric)
            if k not in self.data:
                self.data[k] = [v]
            else:
                self.data[k].append(v)
            if k not in self.epochs:
                self.epochs[k] = [data['epoch']]
            else:
                self.epochs[k].append(data['epoch'])

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def epoch_list(self, k):
        return self.epochs[k]

    def __getitem__(self, k):
        return self.data[k]

    def __contains__(self, k):
        return k in self.data


def parse_args():
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(__file__)
    parser.add_argument('exp', nargs='?', default=None,
                        help='experiment name')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'outputs'))
    parser.add_argument('--cmap', type=str, default='tab10')

    return parser.parse_args()


def main():
    args = parse_args()
    plt.style.use('ggplot')

    output_dir = args.output_dir
    if args.exp is not None:
        output_dir = os.path.join(args.output_dir, args.exp)
    log_file = os.path.join(output_dir, 'log.txt')
    fig_save_dir = os.path.join(output_dir, 'figs')

    m = MetricAggregator()
    with open(log_file, 'r') as f:
        while f.readable():
            line = f.readline()
            if not len(line):
                break
            m.update(json.loads(line))

    cmap = plt.get_cmap(args.cmap)
    parts = ['train', 'val', 'test']
    os.makedirs(fig_save_dir, exist_ok=True)
    for metric in m.metrics:
        print(metric)
        fig, ax = plt.subplots()

        max_epoch = -1
        lines = []
        if (k := metric) in m:
            lines.append(
                ax.plot(el := m.epoch_list(k), m[k], marker='x', color=cmap(0))
            )
            max_epoch = max(el[-1], max_epoch)
        for part in parts:
            if (k := f'{part}_{metric}') in m:
                lines.append(
                    ax.plot(el := m.epoch_list(k), m[k], marker='x', color=cmap(parts.index(part)), label=part)
                )
                max_epoch = max(el[-1], max_epoch)
        plt.minorticks_on()
        ax.grid(True, which='both')
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric)
        ax.set_xlim(0, max_epoch)
        ax.set_ylim(0, ax.get_ylim()[1])
        if len(lines) > 1:
            ax.legend()
        # ax.set_title(metric)

        fig.tight_layout()
        fig.savefig(os.path.join(fig_save_dir, metric + '.png'), transparent=True)
        tikzplotlib.save(os.path.join(fig_save_dir, metric + '.tex'))
        plt.close(fig)


if __name__ == '__main__':
    main()
