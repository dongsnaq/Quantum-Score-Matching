"""Full-circuit simulations from (0.5, 0.5); seed-level errors, no smoothing."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / 'Data/simulation'
output = Path(__file__).resolve().parents[1] / 'Figures'
output.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'mathtext.fontset': 'stixsans', 'font.size': 11, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
target = np.array([1., 1.5])
for beta, color in [(.2, '#3B5BA5'), (.4, '#D47A2C')]:
    data = [json.loads((root / f'beta{beta:.1f}_seed{s}.json').read_text()) for s in range(5)]
    for d in data:
        assert len(d['history']) == 40
        assert d['configuration']['initial'] == [.5, .5]
        assert d['configuration']['circuits'] == 256
    theta = np.array([np.vstack(([.5, .5], [r['theta'] for r in d['history']])) for d in data])
    delta = theta - target
    errors = [np.linalg.norm(delta, axis=2) / np.linalg.norm(target), abs(delta[:, :, 0]) / target[0], abs(delta[:, :, 1]) / target[1]]
    for ax, err in zip(axes, errors):
        mean, sd = err.mean(0), err.std(0, ddof=1)
        ax.plot(np.arange(41), mean, color=color, lw=1.6, label=rf'$\beta={beta:.1f}$')
        ax.fill_between(np.arange(41), mean-sd, mean+sd, color=color, alpha=.13, lw=0)
for ax, title, label in zip(axes, ['(a) Overall', '(b) Coupling $J$', '(c) Transverse field $h$'], [r'Rel. learning error, $e_{\mathrm{rel}}$', r'Rel. error, $|J-J^\ast|/|J^\ast|$', r'Rel. error, $|h-h^\ast|/|h^\ast|$']):
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    ax.set(xlabel='Iteration', ylabel=label, xlim=(0, 40))
    ax.axvline(25, color='.45', ls='--', lw=.9, label='25 iterations', zorder=0)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.grid(ls=':', color='.88', lw=.65)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(.52, .015), frameon=False)
fig.subplots_adjust(left=.065, right=.99, bottom=.25, top=.90, wspace=.30)
fig.savefig(output / 'simulation_learning_errors.pdf', dpi=180, bbox_inches='tight')
