"""Check archived circuits, random instances, mappings, and reconstruction."""
import argparse
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

import qsm_ibm_model as model
import circuits as construction
from analyze import write_csv
from circuit_data import save_circuits, read_arrays, circuit_from_arrays

ROOT = Path(__file__).resolve().parents[1]


def signature(circuit):
    return [(op.operation.name, tuple(float(x) for x in op.operation.params),
        tuple(circuit.find_bit(q).index for q in op.qubits),
        tuple(circuit.find_bit(c).index for c in op.clbits)) for op in circuit.data]


def validate(limit):
    data = ROOT / 'Data'
    cfg = json.loads((data / 'experiment.json').read_text())
    layout = json.loads((data / 'layout.json').read_text())
    owners = {q: i for i, block in enumerate(layout['blocks']) for q in block}
    assert len(owners) == 5*cfg['blocks']
    preparations = construction.prepare_eigenstates()
    save_circuits(data / 'state_preparations.npz', preparations)
    energies, vectors = np.linalg.eigh(np.einsum('j,jab->ab', model.TARGET, model.GENERATORS))
    weights = np.exp(-np.asarray(cfg['betas'])[:, None]*(energies-energies.min()))
    weights /= weights.sum(axis=1, keepdims=True)
    np.savez_compressed(data / 'target_states.npz', eigenvalues=energies, eigenvectors=vectors,
        beta=cfg['betas'], probabilities=weights)
    resource_rows = []
    reconstructed = samples = operations = 0
    for epoch in range(1, limit+1):
        folder = data / f'epoch{epoch:03d}'
        manifest = json.loads((folder / 'instances.json').read_text())
        all_records = sorted([r for pub in manifest['pubs'] for r in pub], key=lambda r: r['instance'])
        assert [r['instance'] for r in all_records] == list(range(2560))
        for bi, beta in enumerate(cfg['betas']):
            for seed in cfg['seeds']:
                key = f'b{beta:g}_seed{seed}'
                state = manifest['groups'][key]
                theta = np.asarray(state['theta_before'])
                rng = np.random.default_rng(np.random.SeedSequence([cfg['sampling_seed_base'], seed, bi, epoch-1]))
                sm, t1, t2 = model.masses(theta, beta, state['cutoff'])
                mass = ((sm+2)[:, None]*(t1+t2)).sum(0)
                np.testing.assert_allclose(mass, state['masses'], atol=1e-12, rtol=0)
                allocated = np.maximum(2, (cfg['instances_per_group']*mass/mass.sum()).astype(int))
                allocated[-1] += cfg['instances_per_group']-allocated.sum()
                assert allocated.tolist() == state['counts']
                for j, count in enumerate(allocated):
                    _, specs = model.draw(theta, beta, state['cutoff'], j, int(count), rng)
                    ks = rng.choice(16, size=count, p=weights[bi])
                    records = [r for r in all_records if r['group'] == key and r['coordinate'] == j]
                    for record, spec, k in zip(records, specs, ks):
                        assert record['spec'] == spec and record['eigenstate'] == int(k)
                        samples += 1
        order = np.random.default_rng(np.random.SeedSequence([cfg['assignment_seed'], epoch-1])).permutation(len(all_records))
        assert [r['instance'] for pub in manifest['pubs'] for r in pub] == order.tolist()
        arrays = read_arrays(folder / 'circuits.npz')
        assert len(arrays['num_qubits']) == len(manifest['pubs']) == 135
        durations = json.loads((folder / 'execution.json').read_text())['estimated_circuit_duration_seconds']
        for pi, records in enumerate(manifest['pubs']):
            assert arrays['num_qubits'][pi] == layout['num_qubits']
            assert arrays['num_clbits'][pi] == len(records)
            lo, hi = arrays['circuit_offsets'][pi:pi+2]
            names = arrays['gate'][lo:hi]
            assert set(names) <= {'rz', 'sx', 'x', 'cz', 'measure'}
            r0, r1 = arrays['register_offsets'][pi:pi+2]
            assert arrays['register_names'][r0:r1].tolist() == [r['register'] for r in records]
            for slot, record in enumerate(records):
                assert record['register'] == f'block{slot}'
                assert record['physical'] == layout['blocks'][slot]
            first, second = arrays['qubit0'][lo:hi], arrays['qubit1'][lo:hi]
            owner_array = np.full(layout['num_qubits'], -1, dtype=int)
            for qubit, owner in owners.items():
                owner_array[qubit] = owner
            assert np.all(owner_array[first] >= 0)
            paired = second >= 0
            assert np.all(owner_array[first[paired]] == owner_array[second[paired]])
            measured = names == 'measure'
            destinations = arrays['clbit'][lo:hi][measured]
            assert sorted(destinations.tolist()) == list(range(len(records)))
            for qubit, ci in zip(first[measured], destinations):
                assert qubit == records[ci]['physical'][records[ci]['ancilla']]
            depth = [0] * layout['num_qubits']
            for q0, q1 in zip(first.tolist(), second.tolist()):
                if q1 < 0:
                    depth[q0] += 1
                else:
                    depth[q0] = depth[q1] = max(depth[q0], depth[q1]) + 1
            operations += hi-lo
            resource_rows.append(dict(epoch=epoch, circuit=pi, blocks=len(records), depth=int(max(depth)),
                gates=int(np.sum(names != 'measure')), cz_gates=int(np.sum(names == 'cz')),
                estimated_duration_seconds=durations[pi]))
        # Reconstruct two independent local instances in every recorded epoch.
        for pi, slot in [(0, 0), (134, len(manifest['pubs'][-1])-1)]:
            circuit = circuit_from_arrays(arrays, pi)
            record = manifest['pubs'][pi][slot]
            small = QuantumCircuit(5, 1)
            positions = {q: i for i, q in enumerate(record['physical'])}
            for op in circuit.data:
                wires = [circuit.find_bit(q).index for q in op.qubits]
                if wires[0] in positions:
                    small.append(op.operation, [positions[q] for q in wires], [0] if op.clbits else [])
            theta = manifest['groups'][record['group']]['theta_before']
            rebuilt, ancilla = construction.build(record['spec'], theta, cfg['dt'], record['eigenstate'], preparations)
            assert ancilla == record['ancilla']
            assert signature(small) == signature(rebuilt), (epoch, pi, slot)
            reconstructed += 1
        print(f'Validated circuits and sampling: {epoch:02d}/{limit}', flush=True)
    if limit == cfg['epochs']:
        write_csv(data / 'circuit_resources.csv', resource_rows)
    report = dict(epochs_checked=limit, circuits_checked=len(resource_rows),
        native_instructions_checked=int(operations), random_instances_reproduced=samples,
        local_circuits_reconstructed=reconstructed, exact_instruction_match=True)
    if limit == cfg['epochs']:
        (data / 'circuit_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=45, choices=range(1, 46))
    validate(parser.parse_args().epochs)
