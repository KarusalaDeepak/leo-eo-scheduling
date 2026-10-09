"""Render completion and paired uncertainty from saved, unchanged experiment data."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl-cache'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def plot_matched_results(frame, evidence, output):
    # Average random seeds within instance first; every scenario gets equal weight.
    grouped = frame.groupby(['stress', 'instance', 'policy']).completion.mean()
    means = grouped.groupby(['stress', 'policy']).mean()
    policies = ['single_margin', 'full_earliest', 'full_margin',
                'full_congestion', 'full_random', 'full_learned', 'full_racs']
    names = ['Single margin', 'Full earliest', 'Full margin',
             'Full congestion', 'Full random', 'Full learned', 'RACS default']
    stresses = ['base', 'tight_deadline', 'reduced_bandwidth', 'high_volume', 'combined']
    values = np.array([[100 * means.loc[s, p] for s in stresses] for p in policies])
    with plt.rc_context({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 1.90),
                                  gridspec_kw={'width_ratios': [1.2, 1]})
        fig.subplots_adjust(left=.135, right=.985, bottom=.21, top=.81, wspace=.60)
        norm = Normalize(0, 100)
        a.imshow(values, cmap='Blues', norm=norm, aspect='auto')
        for row in range(len(policies)):
            for col in range(len(stresses)):
                a.text(col, row, f'{values[row, col]:.1f}', ha='center', va='center',
                       fontsize=7.7, color='white' if values[row, col] > 62 else '#122b40')
        a.set_xticks(range(5), ['Base', 'Tight', 'BW', 'Volume', 'Joint'])
        a.set_yticks(range(7), names)
        a.tick_params(length=0, pad=4)
        a.set_xlabel('Stress condition', labelpad=6)
        a.add_patch(Rectangle((-.5, 5.5), 5, 1, fill=False, edgecolor='#101820', lw=1.2))
        for spine in a.spines.values():
            spine.set_visible(False)

        # Separate the restricted baseline so it cannot compress the matched comparisons.
        b.set_position([.69, .20, .29, .37])
        restricted = fig.add_axes([.69, .71, .29, .11])
        # Use existing paired bootstrap intervals, not independent-policy bars.
        joint = evidence[evidence.stress == 'combined'].set_index('comparison')
        for y, p in enumerate(policies[:-1]):
            axis = restricted if p == 'single_margin' else b
            y = 0 if p == 'single_margin' else y - 1
            r = joint.loc['full_racs vs ' + p]
            lo, mid, hi = 100 * r.ci_low, 100 * r.delta, 100 * r.ci_high
            color = '#00758a' if p == 'single_margin' else '#425466'
            axis.plot([lo, hi], [y, y], color=color, lw=1.5)
            axis.plot([lo, hi], [y, y], '|', color=color, ms=6)
            axis.plot(mid, y, 'o', color=color, ms=4)
        restricted.set_yticks([0], ['Single margin'])
        restricted.set_ylim(-.6, .6)
        restricted.set_xlim(-1, 18)
        restricted.set_xticks([0, 5, 10, 15])
        b.set_yticks(range(5), names[1:-1])
        b.set_ylim(4.6, -.6)
        b.set_xlim(-1.2, 3.2)
        b.set_xticks([-1, 0, 1, 2, 3])
        b.set_xlabel('RACS minus comparator (pp)', fontsize=8, labelpad=6)
        for axis in [b, restricted]:
            axis.axvline(0, color='#222222', ls='--', lw=.8, zorder=0)
            axis.grid(axis='x', color='#dddddd', lw=.5)
            axis.set_axisbelow(True)
            axis.tick_params(axis='y', length=0, labelsize=7.5)
            axis.tick_params(axis='x', labelsize=7)
            for side in ['top', 'right', 'left']:
                axis.spines[side].set_visible(False)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for ext in ['pdf', 'png']:
            fig.savefig(output / f'matched_results.{ext}', dpi=400,
                        bbox_inches='tight', pad_inches=.02)
        plt.close(fig)


if __name__ == '__main__':
    source = ROOT / 'results/reviewer_revision'
    plot_matched_results(pd.read_csv(source / 'instance_results.csv'),
                         pd.read_csv(source / 'paired_statistics.csv'),
                         ROOT / 'revision/figures')
