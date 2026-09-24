"""Recompute every gradient and optimizer state from the hardware bitstrings."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def update(theta, gradient, beta, iteration, cfg):
    rate = cfg['initial_learning_rate'] / (1 + iteration / cfg['learning_rate_time_scale'])
    step = -rate * gradient / (beta**2 * np.asarray(cfg['gram_diagonal']))
    cap = cfg['step_norm_cap_fraction'] * np.linalg.norm(cfg['target'])
    if np.linalg.norm(step) > cap:
        step *= cap / np.linalg.norm(step)
    return np.clip(theta + step, *cfg['parameter_bounds'])


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    data = ROOT / 'Data'
    cfg = json.loads((data / 'experiment.json').read_text())
    nb, ns, nt = len(cfg['betas']), len(cfg['seeds']), cfg['epochs']
    theta = np.zeros((nb, ns, nt + 1, 2))
    gradients = np.zeros((nb, ns, nt, 2))
    masses = np.zeros_like(gradients)
    instance_counts = np.zeros_like(gradients, dtype=int)
    theta[:, :, 0] = cfg['initial']
    target = np.asarray(cfg['target'])
    rows, summary, total_outcomes = [], [], 0
    max_gradient_difference = max_theta_difference = 0.
    for bi, beta in enumerate(cfg['betas']):
        for si, seed in enumerate(cfg['seeds']):
            delta = np.asarray(cfg['initial']) - target
            rows.append(dict(beta=beta, seed=seed, epoch=0, J=cfg['initial'][0], h=cfg['initial'][1],
                gradient_J='', gradient_h='', mass_J='', mass_h='', instances_J='', instances_h='',
                learning_rate='', cumulative_measurements=0,
                error_relative=float(np.linalg.norm(delta) / np.linalg.norm(target)),
                error_J=float(abs(delta[0]) / abs(target[0])), error_h=float(abs(delta[1]) / abs(target[1]))))
    for t in range(nt):
        folder = data / f'epoch{t+1:03d}'
        manifest = json.loads((folder / 'instances.json').read_text())
        saved = json.loads((folder / 'optimization.json').read_text())
        counts = json.loads((folder / 'counts.json').read_text())
        assert manifest['epoch'] == saved['epoch'] == t
        assert len(counts) == len(manifest['pubs'])
        values = {key: [[], []] for key in manifest['groups']}
        expected_keys = set()
        with np.load(folder / 'outcomes.npz', allow_pickle=False) as outcomes:
            for pi, (pub, records) in enumerate(zip(counts, manifest['pubs'])):
                assert set(pub) == {r['register'] for r in records}
                for record in records:
                    key = f"pub{pi:03d}_{record['register']}"
                    expected_keys.add(key)
                    bits = outcomes[key]
                    assert bits.shape == (manifest['shots'], 1)
                    assert bits.dtype == np.uint8 and np.all((bits == 0) | (bits == 1))
                    n1 = int(bits.sum())
                    n0 = len(bits) - n1
                    measured = pub[record['register']]
                    assert measured.get('0', 0) == n0 and measured.get('1', 0) == n1
                    values[record['group']][record['coordinate']].append(record['sign'] * (n0-n1) / len(bits))
                    assert record['sign'] == record['spec']['sign']
                    total_outcomes += len(bits)
            assert set(outcomes.files) == expected_keys
        for bi, beta in enumerate(cfg['betas']):
            for si, seed in enumerate(cfg['seeds']):
                key = f'b{beta:g}_seed{seed}'
                state = manifest['groups'][key]
                reference = saved['groups'][key]
                assert [len(v) for v in values[key]] == state['counts']
                assert sum(state['counts']) == cfg['instances_per_group']
                np.testing.assert_allclose(theta[bi, si, t], state['theta_before'], atol=1e-12, rtol=0)
                grad = np.asarray([np.mean(v) for v in values[key]]) * state['masses']
                after = update(theta[bi, si, t], grad, beta, t, cfg)
                np.testing.assert_allclose(grad, reference['gradient'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(after, reference['theta'], atol=1e-12, rtol=0)
                max_gradient_difference = max(max_gradient_difference, float(np.max(abs(grad-reference['gradient']))))
                max_theta_difference = max(max_theta_difference, float(np.max(abs(after-reference['theta']))))
                theta[bi, si, t+1] = after
                gradients[bi, si, t] = grad
                masses[bi, si, t] = state['masses']
                instance_counts[bi, si, t] = state['counts']
                delta = after-target
                error = np.linalg.norm(delta) / np.linalg.norm(target)
                assert abs(error-reference['error']) < 1e-12
                rows.append(dict(beta=beta, seed=seed, epoch=t+1, J=after[0], h=after[1],
                    gradient_J=grad[0], gradient_h=grad[1], mass_J=state['masses'][0], mass_h=state['masses'][1],
                    instances_J=state['counts'][0], instances_h=state['counts'][1],
                    learning_rate=cfg['initial_learning_rate']/(1+t/cfg['learning_rate_time_scale']),
                    cumulative_measurements=(t+1)*cfg['instances_per_group']*cfg['shots_per_instance'],
                    error_relative=error, error_J=abs(delta[0])/abs(target[0]), error_h=abs(delta[1])/abs(target[1])))
        print(f'Validated measurements and updates: {t+1:02d}/{nt}', flush=True)
    np.savez_compressed(data / 'trajectories.npz', theta=theta, gradient=gradients, coefficient_mass=masses,
        instance_counts=instance_counts, beta=cfg['betas'], seed=cfg['seeds'], target=target, initial=cfg['initial'])
    write_csv(data / 'trajectories.csv', sorted(rows, key=lambda r: (r['beta'], r['seed'], r['epoch'])))
    for bi, beta in enumerate(cfg['betas']):
        delta = theta[bi, :, -1] - target
        for name, err in [('overall', np.linalg.norm(delta, axis=-1)/np.linalg.norm(target)),
                          ('J', abs(delta[:, 0])/abs(target[0])), ('h', abs(delta[:, 1])/abs(target[1]))]:
            summary.append(dict(beta=beta, parameter=name, mean_relative_error=float(err.mean()),
                sample_standard_deviation=float(err.std(ddof=1)), trajectories=ns, epochs=nt))
    write_csv(data / 'final_errors.csv', summary)
    execution = json.loads((data / 'execution_summary.json').read_text())
    report = dict(epochs=nt, trajectories=nb*ns, circuits=nt*135, random_instances=nt*nb*ns*cfg['instances_per_group'],
        measured_bits=total_outcomes, measurements_per_trajectory=nt*cfg['instances_per_group']*cfg['shots_per_instance'],
        quantum_seconds=sum(r['usage']['quantum_seconds'] for r in execution),
        max_gradient_reconstruction_error=max_gradient_difference, max_parameter_reconstruction_error=max_theta_difference)
    (data / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    analyze()
