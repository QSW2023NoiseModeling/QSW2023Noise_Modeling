# SPDX-License-Identifier: Apache-2.0

# (C) Copyright XXX 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# We honor the previous contributions and publish own contributions 
# under the same Apache 2.0 License, to the best of our knowldedge, in compliance with the Apache-2.0 license terms.

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT



def init_circuit(nqubits: int):
    """
    Initializes superposition of the circuit with n-qubits
    """
    circuit = QuantumCircuit(nqubits)
    for q in range(nqubits):
        circuit.h(q)
    return circuit

def grover_oracle(nqubits: int, pos_winner: int):
    """
    Calculates Grover oracle for single solution (bzw. flips amplitude of winner)
    """
    circuit = QuantumCircuit(nqubits)
    
    # Flip Bitstring for better looping
    bin_str = np.binary_repr(pos_winner, nqubits)
    bin_str = bin_str[::-1]

    # Set all zero-entries in bitstring to X
    for pos, num in enumerate(bin_str):
        if num == '0':
            circuit.x(pos)

    multi_z = MCMT('z', nqubits-1, 1)
    circuit = circuit.compose(multi_z)

    # Set all zero-entries in bitstring to X
    for pos, num in enumerate(bin_str):
        if num == '0':
            circuit.x(pos)

    return circuit


def grover_turner(nqubits: int):
    """
    Grover iterates ~sqrt(N) times
    Total Turning angle = 2*theta with theta = arcsin(1/sqrt(N))

    |w>
    |
    |
    |          x  |s>  
    |        x
    |      x
    |    x
    |  x     theta
    |x
    ------------------------> |s'>
    """
   
    # Transform |s> --> |0..>
    circuit = QuantumCircuit(nqubits)
    for q in range(nqubits):
        circuit.h(q)
        circuit.x(q)

    multi_z = MCMT('z', nqubits-1, 1)
    circuit = circuit.compose(multi_z)

    for q in range(nqubits):
        circuit.x(q)
        circuit.h(q)

    return circuit

def get_grover_circuit(run_config: dict):
    """
    Compose complete grover circuit for correct number of iters
    """
    nqubits = run_config["nqubits"]
    winner = run_config["winner_state"]

    N = 2**nqubits
    iters = int(np.round(np.pi /(4*np.arcsin(1/np.sqrt(N))) - 1/2))
    circuit = init_circuit(nqubits)
    oracle = grover_oracle(nqubits,winner)
    turner = grover_turner(nqubits)

    for _ in range(iters):
        circuit = circuit.compose(oracle)
        circuit = circuit.compose(turner)
    return circuit