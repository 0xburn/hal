import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('/tmp/hal-rank1-o59-calibration')
SUMMARY = json.loads(Path('outputs/o59-rank1-return-calibration.json').read_text())
GAMES = [json.loads((ROOT/'rows'/f'{i:03d}.json').read_text()) for i in range(200)]
TEST = [row for game in GAMES if game['set']=='test' for row in game['rows'] if row['base']]
TAIL = [row for game in GAMES for row in game['rows'] if row['tail']]


def main():
    value = np.asarray([row['value'] for row in TEST]); ret = np.asarray([row['return60'] for row in TEST])
    tail_value = np.asarray([row['value'] for row in TAIL]); tail_ret = np.asarray([row['return60'] for row in TAIL])
    intercept, slope50 = SUMMARY['models']['affine']['coefficients']
    grid = np.linspace(value.min(), value.max(), 200)
    fit = intercept + slope50*grid/50
    fig, axes = plt.subplots(3, 1, figsize=(8.3, 12), constrained_layout=True)
    fig.suptitle('O59 value vs realized 60-frame return · ranked-anon-1', fontsize=15)
    ax = axes[0]
    ax.scatter(value, ret, s=13, alpha=.34, color='#4b728e', edgecolors='none')
    ax.scatter(value[ret>100],ret[ret>100],s=27,alpha=.9,color='#cd6b36',edgecolors='none',label='Return >100')
    ax.plot(grid,fit,color='#573d94',linewidth=2,label='Fitted 80th percentile')
    ax.axhline(0,color='#555555',linewidth=.7,alpha=.5)
    ax.set(xlabel='Value head prediction (return points)',ylabel='Realized 60-frame return (points)',title=f"Representative frames · 50 held-out games · Pearson r = {SUMMARY['summary']['test']['frame_weighted_pearson']:.3f}")
    ax.legend(loc='upper left',frameon=False)

    ax = axes[1]
    center=(ret>=-25)&(ret<=25)
    ax.scatter(value[center],ret[center],s=13,alpha=.27,color='#4b728e',edgecolors='none')
    ax.plot(grid,fit,color='#573d94',linewidth=2,label='Fitted 80th percentile')
    ax.axhline(0,color='#555555',linewidth=.7,alpha=.5)
    ax.axvspan(0,20,color='#d4b051',alpha=.17)
    ax.set(xlim=(-80,100),ylim=(-25,25),xlabel='Value head prediction (return points)',ylabel='Realized 60-frame return (points)',title='Typical-return range · shaded: value from 0 to 20')
    ax.legend(loc='upper left',frameon=False)

    ax = axes[2]
    ax.scatter(tail_value,tail_ret,s=11,alpha=.20,color='#cd6b36',edgecolors='none')
    near=(tail_value>=0)&(tail_value<20)
    ax.scatter(tail_value[near],tail_ret[near],s=18,alpha=.7,color='#573d94',edgecolors='none',label=f'{near.sum()} samples in value 0–20')
    ax.axvspan(0,20,color='#d4b051',alpha=.17)
    ax.axhline(100,color='#555555',linewidth=.7,alpha=.5)
    ax.set(xlabel='Value head prediction (return points)',ylabel='Realized 60-frame return (points)',title='>100-return windows · enriched sample, excluded from fit')
    ax.legend(loc='upper left',frameon=False)
    for axis in axes:
        axis.grid(alpha=.15)
        axis.spines[['top','right']].set_visible(False)
    for extension in ('png','pdf'):
        out=Path(f'outputs/o59-rank1-return-calibration.{extension}')
        fig.savefig(out,dpi=180)
        print(out.resolve())
    plt.close(fig)


if __name__=='__main__':
    main()
