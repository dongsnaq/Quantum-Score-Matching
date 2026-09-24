"""Offline native-circuit training with the saved IBM calibration."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.models import BackendProperties

import circuits
import qsm_ibm_model as model

ROOT = Path(__file__).resolve().parents[1]
PATCH = [83, 84, 85, 86, 87]


class ZeroTemperatureProperties(BackendProperties):
    def frequency(self, qubit):
        # The archived calibration omits frequencies. At zero bath temperature,
        # Aer uses zero thermal excited-state population independently of them.
        return 0.0


def initialize_worker(calibration, preparations, ideal):
    global SIMULATOR, PREPARATIONS
    PREPARATIONS = preparations
    if ideal:
        noise = None
    else:
        data = json.loads(Path(calibration).read_text())
        indices = {physical: local for local, physical in enumerate(PATCH)}
        selected = dict(data)
        selected['qubits'] = [data['qubits'][q] for q in PATCH]
        selected['gates'] = [dict(g, qubits=[indices[q] for q in g['qubits']])
                             for g in data['gates'] if all(q in indices for q in g['qubits'])]
        properties = ZeroTemperatureProperties.from_dict(selected)
        noise = NoiseModel.from_backend_properties(properties, temperature=0)
    SIMULATOR = AerSimulator(method='density_matrix', noise_model=noise, max_parallel_threads=1)


def measure_instance(item):
    spec, theta, dt, eigenstate, repetitions, seed = item
    circuit, _ = circuits.build(spec, theta, dt, eigenstate, PREPARATIONS)
    counts = SIMULATOR.run(circuit, shots=repetitions, seed_simulator=int(seed)).result().get_counts()
    n0, n1 = int(counts.get('0', 0)), int(counts.get('1', 0))
    assert n0 + n1 == repetitions
    return {
        'counts': [n0, n1], 'value': spec['sign'] * (n0-n1)/repetitions,
        'cz': int(circuit.count_ops().get('cz', 0)), 'depth': circuit.depth(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--circuits', type=int, default=256)
    parser.add_argument('--shots', type=int, default=8192)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--initial', type=float, nargs=2, default=[0.5, 0.5])
    parser.add_argument('--dt', type=float, default=0.25)
    parser.add_argument('--ideal', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--calibration', type=Path, default=ROOT/'Data/simulation/calibration.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.circuits < 4 or args.shots < args.circuits or args.shots % args.circuits:
        raise ValueError('Use at least four instances and a shot budget divisible by the instance count')
    repetitions = args.shots // args.circuits
    configurations = {k: getattr(args, k) for k in
                      ('beta', 'circuits', 'shots', 'seed', 'initial', 'dt', 'ideal')}
    calibration = json.loads(args.calibration.read_text())
    configurations['calibration_date'] = str(calibration['last_update_date'])
    preparations = circuits.prepare_eigenstates()
    energies, _ = np.linalg.eigh(np.einsum('j,jab->ab', model.TARGET, model.GENERATORS))
    weights = np.exp(-args.beta*(energies-energies.min()))
    weights /= weights.sum()
    b = np.array([sum(abs(c) for c, _ in model.commutator_terms(i, np.array([2., 2.]))) for i in range(8)])
    q = np.array([[sum(abs(c) for c, _ in model.commutator_terms(i, coordinate=j)) for j in range(2)] for i in range(8)])
    cutoff = model.uniform_cutoff(args.beta, 1e-4, b, q, np.array([3., 4.]))
    streams = np.random.SeedSequence([93020, args.seed]).spawn(2)
    rng, measurement_rng = [np.random.default_rng(stream) for stream in streams]
    theta = np.array(args.initial)
    history = []
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        saved = json.loads(args.output.read_text())
        if saved['configuration'] != configurations:
            raise ValueError('Resume configuration differs from the saved simulation')
        history = saved['history']
        theta = np.asarray(history[-1]['theta'])
        rng.bit_generator.state = saved['circuit_rng']
        measurement_rng.bit_generator.state = saved['measurement_rng']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers, mp_context=get_context('spawn'),
            initializer=initialize_worker,
            initargs=(str(args.calibration), preparations, args.ideal)) as pool:
        for t in range(len(history), args.steps):
            sm, t1, t2 = model.masses(theta, args.beta, cutoff)
            masses = ((sm+2)[:, None]*(t1+t2)).sum(0)
            allocation = np.maximum(2, (args.circuits*masses/masses.sum()).astype(int))
            allocation[-1] += args.circuits-allocation.sum()
            gradient, coordinates = [], []
            for j, count in enumerate(allocation):
                mass, specs = model.draw(theta, args.beta, cutoff, j, int(count), rng)
                eigenstates = rng.choice(16, size=count, p=weights)
                seeds = measurement_rng.integers(1, 2**31-1, size=count)
                items = [(s, theta, args.dt, int(k), repetitions, int(seed))
                         for s, k, seed in zip(specs, eigenstates, seeds)]
                results = list(pool.map(measure_instance, items, chunksize=4))
                gradient.append(float(mass*np.mean([r['value'] for r in results])))
                coordinates.append({'coordinate': j, 'mass': mass, 'instances': int(count),
                    'specs': specs, 'eigenstates': eigenstates.tolist(), 'outcomes': results})
            updated = model.update(theta, np.asarray(gradient), args.beta, t)
            history.append({'iteration': t, 'theta_before': theta.tolist(), 'theta': updated.tolist(),
                'measurement_gradient': gradient, 'coordinates': coordinates,
                'error': float(np.linalg.norm(updated-model.TARGET)/np.linalg.norm(model.TARGET))})
            theta = updated
            result = {'configuration': configurations, 'steps_completed': len(history),
                'repetitions_per_circuit': repetitions, 'cutoff': cutoff, 'history': history,
                'circuit_rng': rng.bit_generator.state, 'measurement_rng': measurement_rng.bit_generator.state}
            temporary = args.output.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, indent=2)+'\n')
            temporary.replace(args.output)
            print(f'iteration {t+1}: relative error {history[-1]["error"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
