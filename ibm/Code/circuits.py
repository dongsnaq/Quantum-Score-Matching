"""Construct native five-qubit measurement circuits and pack physical blocks."""
import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation, Optimize1qGatesDecomposition

import qsm_ibm_model as model
from circuit_data import save_circuits

ROOT = Path(__file__).resolve().parents[1]
BASIS = ['rz', 'sx', 'x', 'cz']


def compile_routed(circuit):
    settings = json.loads((ROOT / 'Data/compiler.json').read_text())
    routed = transpile(circuit, basis_gates=BASIS, coupling_map=settings['coupling_map'],
        initial_layout=list(range(5)), optimization_level=1, seed_transpiler=31, approximation_degree=1.0)
    positions = list(routed.layout.final_index_layout())
    routed = PassManager([CommutativeCancellation(basis_gates=BASIS),
        Optimize1qGatesDecomposition(basis=BASIS)]).run(routed)
    return routed, positions


def prepare_eigenstates():
    """Prepare target eigenvectors, restoring all five logical output wires."""
    _, vectors = np.linalg.eigh(np.einsum('j,jab->ab', model.TARGET, model.GENERATORS))
    settings = json.loads((ROOT / 'Data/compiler.json').read_text())
    neighbors = {i: set() for i in range(5)}
    for a, b in settings['coupling_map']:
        neighbors[a].add(b)
        neighbors[b].add(a)
    preparations = []
    for k in range(16):
        circuit = QuantumCircuit(5)
        circuit.append(StatePreparation(vectors[:, k]), [1, 2, 3, 4])
        compiled, positions = compile_routed(circuit)
        restored = QuantumCircuit(5)
        restored.compose(compiled, inplace=True)
        for logical in range(5):
            a, b = positions[logical], logical
            if a == b:
                continue
            queue = deque([[a]])
            while queue:
                path = queue.popleft()
                if path[-1] == b:
                    break
                queue.extend(path + [v] for v in sorted(neighbors[path[-1]]) if v not in path)
            edges = list(zip(path[:-1], path[1:]))
            for u, v in edges + edges[-2::-1]:
                restored.swap(u, v)
            positions = [b if p == a else a if p == b else p for p in positions]
        restored = transpile(restored, basis_gates=BASIS, optimization_level=1, approximation_degree=1.0)
        actual = Statevector.from_instruction(restored).data
        expected = np.kron(vectors[:, k], [1., 0.])
        assert abs(abs(np.vdot(expected, actual)) - 1) < 1e-10
        preparations.append(restored)
    return preparations


def evolution(time, theta, dt):
    """Second-order product formula for exp(+i time H)."""
    circuit = QuantumCircuit(4)
    steps = max(1, int(np.ceil(abs(time) / dt)))
    delta = time / steps
    for i in range(3):
        circuit.rzz(theta[0] * delta, i, i+1)
    for step in range(steps):
        for i in range(4):
            circuit.rx(2 * theta[1] * delta, i)
        for i in range(3):
            circuit.rzz(theta[0] * delta * (1 if step == steps-1 else 2), i, i+1)
    return circuit


def conjugated_pauli(time, pauli, theta, dt, s=None, insertion=None, eta=1):
    if s is None:
        return evolution(time, theta, dt), pauli
    circuit = evolution(s * time, theta, dt)
    rotation = QuantumCircuit(4)
    occupied = [(i, c) for i, c in enumerate(insertion[::-1]) if c != 'I']
    if len(occupied) == 1:
        rotation.rx(-eta * np.pi/2, occupied[0][0])
    else:
        rotation.rzz(-eta * np.pi/2, occupied[0][0], occupied[1][0])
    circuit.compose(rotation, inplace=True)
    circuit.compose(evolution((1-s) * time, theta, dt), inplace=True)
    return circuit, pauli


def controlled_observable(circuit, unitary, pauli):
    circuit.compose(unitary.inverse(), qubits=[1, 2, 3, 4], inplace=True)
    for i, letter in enumerate(pauli[::-1]):
        if letter == 'X':
            circuit.cx(0, i+1)
        elif letter == 'Y':
            circuit.cy(0, i+1)
        elif letter == 'Z':
            circuit.cz(0, i+1)
    circuit.compose(unitary, qubits=[1, 2, 3, 4], inplace=True)


def build(spec, theta, dt, eigenstate, preparations):
    unitary, pauli = conjugated_pauli(spec['xi'], spec['q'], theta, dt,
        spec['s'], spec['insertion'], spec['eta'])
    measurement = QuantumCircuit(5)
    measurement.h(0)
    controlled_observable(measurement, unitary, pauli)
    if spec['product']:
        companion, pauli = conjugated_pauli(spec['xs'], spec['qs'], theta, dt)
    else:
        companion, pauli = QuantumCircuit(4), spec['qs']
    controlled_observable(measurement, companion, pauli)
    if not spec['product']:
        measurement.sdg(0)
    measurement.h(0)
    routed, positions = compile_routed(measurement)
    circuit = QuantumCircuit(5, 1)
    circuit.compose(preparations[eigenstate], inplace=True)
    circuit.compose(routed, inplace=True)
    circuit.measure(positions[0], 0)
    return circuit, positions[0]


def rebuild_epoch(epoch):
    cfg = json.loads((ROOT / 'Data/experiment.json').read_text())
    layout = json.loads((ROOT / 'Data/layout.json').read_text())
    manifest = json.loads((ROOT / f'Data/epoch{epoch:03d}/instances.json').read_text())
    preparations = prepare_eigenstates()
    circuits = []
    for records in manifest['pubs']:
        circuit = QuantumCircuit(layout['num_qubits'])
        for record in records:
            theta = manifest['groups'][record['group']]['theta_before']
            small, ancilla = build(record['spec'], theta, cfg['dt'], record['eigenstate'], preparations)
            assert ancilla == record['ancilla']
            register = ClassicalRegister(1, record['register'])
            circuit.add_register(register)
            circuit.compose(small, qubits=record['physical'], clbits=list(register), inplace=True)
        circuits.append(circuit)
    return circuits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct recorded circuits offline; no hardware submission.')
    parser.add_argument('--epoch', type=int, required=True, choices=range(1, 46))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.suffix != '.npz':
        raise ValueError('Use an .npz output filename')
    save_circuits(args.output, rebuild_epoch(args.epoch))
