"""Reproduce hardware learning curves on the noiseless population landscape."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Figures'


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', dpi=180, bbox_inches='tight')


def errors(cfg, theta):
    target = np.asarray(cfg['target'])
    delta = theta-target
    errors = [np.linalg.norm(delta, axis=-1)/np.linalg.norm(target),
              abs(delta[:, :, :, 0])/target[0], abs(delta[:, :, :, 1])/target[1]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for i, (ax, err, title) in enumerate(zip(axes, errors, ['(a) Overall', '(b) Coupling $J$', '(c) Field $h$'])):
        for bi, color in enumerate(['#3B5BA5', '#D47A2C']):
            mean, sd = err[bi].mean(0), err[bi].std(0, ddof=1)
            ax.plot(np.arange(theta.shape[2]), mean, color=color, lw=1.6, label=rf'$\beta={cfg["betas"][bi]:.1f}$')
            ax.fill_between(np.arange(theta.shape[2]), mean-sd, mean+sd, color=color, alpha=.13, lw=0)
        ax.set_title(title, loc='left', fontsize=14, fontweight='bold')
        ax.set_xlim(0, cfg['epochs'])
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel([r'Rel. $\ell^2$ learning error',
            r'Rel. error, $|J-J^*|/|J^*|$', r'Rel. error, $|h-h^*|/|h^*|$'][i], fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(ls=':', color='.88', lw=.65)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
        bbox_to_anchor=(.52, .005), frameon=True, fancybox=False,
        framealpha=1., edgecolor='.75', fontsize=12.5)
    fig.subplots_adjust(left=.08, right=.99, bottom=.29, top=.88, wspace=.44)
    save(fig, 'hardware_learning_errors')
    plt.close(fig)


def compute_landscape(cfg):
    import qsm_ibm_model as model
    js, hs = np.linspace(.4, 1.3, 121), np.linspace(.35, 2., 151)
    energy, vectors = np.linalg.eigh(np.einsum('j,jab->ab', cfg['target'], model.GENERATORS))
    grids = []
    for beta in cfg['betas']:
        weights = np.exp(-beta*(energy-energy.min()))
        weights /= weights.sum()
        rho = (vectors*weights) @ vectors.conj().T
        reference = model.loss(np.asarray(cfg['target']), beta, rho)
        grid = np.array([[model.loss(np.array([j, h]), beta, rho)-reference for j in js] for h in hs])
        assert grid.min() > -1e-10
        grids.append(grid)
    np.savez_compressed(ROOT / 'Data/population_landscape.npz', J=js, h=hs, gap=grids, beta=cfg['betas'])


def backdrop(cfg):
    with np.load(ROOT / 'Data/population_landscape.npz') as data:
        gaps, js, hs = data['gap'], data['J'], data['h']
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), sharex=True, sharey=True)
    logs = np.log10(np.maximum(gaps, 1e-7))
    levels = np.linspace(logs.min(), logs.max(), 100)
    for bi, ax in enumerate(axes):
        image = ax.contourf(js, hs, logs[bi], levels=levels, cmap='viridis')
        for collection in image.collections:
            collection.set_edgecolor('face')
        ax.plot(*cfg['target'], marker='*', ms=10, color='white', mec='.15', mew=.7, zorder=5)
        ax.plot(*cfg['initial'], marker='o', ms=4, color='.15', zorder=5)
        ax.set(xlim=(js[0], js[-1]), ylim=(hs[0], hs[-1]))
        ax.set_xlabel(r'Coupling, $J$', fontsize=17)
        ax.tick_params(labelsize=15)
        ax.set_xticks([.4, .8, 1.2])
        ax.set_title(rf'({chr(97+bi)}) $\beta={cfg["betas"][bi]:.1f}$', loc='left', fontsize=17, fontweight='bold')
    axes[0].set_ylabel(r'Field, $h$', fontsize=17)
    fig.subplots_adjust(left=.105, right=.80, bottom=.20, top=.87, wspace=.16)
    color_axis = fig.add_axes([.84, .20, .025, .67])
    bar = fig.colorbar(image, cax=color_axis, orientation='vertical')
    bar.set_ticks(np.arange(np.ceil(logs.min()), np.floor(logs.max())+1))
    bar.ax.tick_params(labelsize=13)
    bar.set_label(r'$\log_{10}[J_Q(\theta)-J_Q(\theta^*)]$', fontsize=16, labelpad=8)
    return fig, axes


def trajectories(cfg, theta, animate):
    colors = ['#E4572E', '#111111', '#E889C2', '#FFFFFF', '#83D7ED']
    fig, axes = backdrop(cfg)
    for bi, ax in enumerate(axes):
        for si, color in enumerate(colors):
            path = theta[bi, si]
            ax.plot(path[:, 0], path[:, 1], color=color, lw=1.05, alpha=.95)
    handles = [Line2D([0], [0], color=c, lw=1.5, label=f'Seed {s}') for s, c in enumerate(colors)]
    handles[3].set_path_effects([pe.Stroke(linewidth=2.5, foreground='.5'), pe.Normal()])
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.46, 1.), ncol=5, frameon=False)
    save(fig, 'hardware_seed_trajectories')
    plt.close(fig)
    fig, axes = backdrop(cfg)
    means = theta.mean(1)
    lines, dots = [], []
    for ax in axes:
        line, = ax.plot([], [], color='#E4572E', lw=1.8)
        dot, = ax.plot([], [], color='#E4572E', marker='o', ms=4)
        lines.append(line)
        dots.append(dot)
    title = fig.text(.46, .96, '', ha='center', va='center', fontsize=12)

    def update(epoch):
        title.set_text(f'Epoch {epoch} / {cfg["epochs"]}')
        for bi, (line, dot) in enumerate(zip(lines, dots)):
            path = means[bi, :epoch+1]
            line.set_data(path[:, 0], path[:, 1])
            dot.set_data([path[-1, 0]], [path[-1, 1]])
        return [title, *lines, *dots]

    if animate:
        animation = FuncAnimation(fig, update, frames=list(range(cfg['epochs']+1))+[cfg['epochs']]*8,
            interval=180, blit=False)
        animation.save(OUT / 'hardware_mean_dynamics.gif', writer=PillowWriter(fps=6), dpi=130)
    update(cfg['epochs'])
    title.set_visible(False)
    save(fig, 'hardware_mean_dynamics')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute-landscape', action='store_true')
    parser.add_argument('--gif', action='store_true')
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'stixsans', 'font.size': 11, 'pdf.fonttype': 42, 'axes.linewidth': .85})
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((ROOT / 'Data/experiment.json').read_text())
    with np.load(ROOT / 'Data/trajectories.npz') as data:
        theta = data['theta']
    if args.recompute_landscape:
        compute_landscape(cfg)
    errors(cfg, theta)
    trajectories(cfg, theta, args.gif)
