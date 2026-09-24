"""Plain NumPy storage of the native circuit gate lists (no pickled objects)."""
import numpy as np


def save_circuits(path, circuits):
    names, first, second, classical, angles = [], [], [], [], []
    offsets, qubits, bits, phases = [0], [], [], []
    register_offsets, register_names = [0], []
    for circuit in circuits:
        qubits.append(circuit.num_qubits)
        bits.append(circuit.num_clbits)
        phases.append(float(circuit.global_phase))
        assert not circuit.parameters and not circuit.metadata
        for register in circuit.cregs:
            assert len(register) == 1
            register_names.append(register.name)
        register_offsets.append(len(register_names))
        for op in circuit.data:
            name = op.operation.name
            assert name in {'rz', 'sx', 'x', 'cz', 'measure'}
            wires = [circuit.find_bit(q).index for q in op.qubits]
            names.append(name)
            first.append(wires[0])
            second.append(wires[1] if len(wires) == 2 else -1)
            classical.append(circuit.find_bit(op.clbits[0]).index if op.clbits else -1)
            angles.append(float(op.operation.params[0]) if name == 'rz' else 0.)
        offsets.append(len(names))
    np.savez_compressed(path, gate=np.asarray(names, dtype='U7'), qubit0=np.asarray(first, dtype=np.int16),
        qubit1=np.asarray(second, dtype=np.int16), clbit=np.asarray(classical, dtype=np.int16),
        angle=np.asarray(angles, dtype=np.float64), circuit_offsets=np.asarray(offsets, dtype=np.int64),
        num_qubits=np.asarray(qubits, dtype=np.int16), num_clbits=np.asarray(bits, dtype=np.int16),
        global_phase=np.asarray(phases, dtype=np.float64), register_offsets=np.asarray(register_offsets, dtype=np.int64),
        register_names=np.asarray(register_names, dtype='U20'))


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def circuit_from_arrays(data, index):
    from qiskit import QuantumCircuit, ClassicalRegister
    circuit = QuantumCircuit(int(data['num_qubits'][index]))
    r0, r1 = data['register_offsets'][index:index+2]
    for name in data['register_names'][r0:r1]:
        circuit.add_register(ClassicalRegister(1, str(name)))
    circuit.global_phase = float(data['global_phase'][index])
    start, end = data['circuit_offsets'][index:index+2]
    for k in range(start, end):
        name, q = str(data['gate'][k]), int(data['qubit0'][k])
        if name == 'rz':
            circuit.rz(float(data['angle'][k]), q)
        elif name == 'sx':
            circuit.sx(q)
        elif name == 'x':
            circuit.x(q)
        elif name == 'cz':
            circuit.cz(q, int(data['qubit1'][k]))
        elif name == 'measure':
            circuit.measure(q, int(data['clbit'][k]))
        else:
            raise ValueError(name)
    return circuit


def load_circuit(path, index=0):
    return circuit_from_arrays(read_arrays(path), index)
