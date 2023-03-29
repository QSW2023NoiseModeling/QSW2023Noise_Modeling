# SPDX-License-Identifier: Apache-2.0

# (C) Copyright IBM 2017.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# (C) Copyright XXX 2023.

# NOTE: In this file the following methods use derivations of source code from the Qiskit project.
# 1) estimate_circuit_time: 
# - The method is based on Qiskits depth function (See: qiskit/circuit/quantumcircuit.py) We altered the method such that it estimates the time
# of a circuit bath based on average 1 and 2 Qubit gate times instead. 

# We honor the previous contributions and publish own contributions 
# under the same Apache 2.0 License, to the best of our knowldedge, in compliance with the Apache-2.0 license terms.


import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister
from grover_lib import*
from typing import Optional
from qiskit.circuit import Clbit, Parameter
from qiskit_aer.backends import AerSimulator
from qiskit.circuit.library import QFT
from qiskit.converters import circuit_to_dag, dag_to_circuit
from collections import OrderedDict
import pathlib
import os
import pickle
import networkx as nx
import itertools

def estimate_circuit_time(circ,tg1,tg2,
        filter_function: Optional[callable] = lambda x: not getattr(
            x.operation, "_directive", False
        ),
    ) -> int:
        """
        Complement qiskit depth function by time estimate
        NOTE: Lambda functions for rz, z, since we only want to investigate on gate elements with runtime, negleting measurement time
        """
        # Assign each bit in the circuit a unique integer
        # to index into op_stack.
        bit_indices = {bit: idx for idx, bit in enumerate(circ.qubits + circ.clbits)}

        # If no bits, return 0
        if not bit_indices:
            return 0

        op_stack = [0] * len(bit_indices)

        time_estimate = 0
        multi_qgate_layer = [False] # Check if current layer has multi (2Q Bit) gate
        # End Edits

        for instruction in circ._data:
            levels = []
            reg_ints = []
            for ind, reg in enumerate(instruction.qubits + instruction.clbits):
                reg_ints.append(bit_indices[reg])
                if filter_function(instruction):
                    levels.append(op_stack[reg_ints[ind]] + 1)
                else:
                    levels.append(op_stack[reg_ints[ind]])
            if getattr(instruction.operation, "condition", None):
                if isinstance(instruction.operation.condition[0], Clbit):
                    condition_bits = [instruction.operation.condition[0]]
                else:
                    condition_bits = instruction.operation.condition[0]
                for cbit in condition_bits:
                    idx = bit_indices[cbit]
                    if idx not in reg_ints:
                        reg_ints.append(idx)
                        levels.append(op_stack[idx] + 1)

            max_level = max(levels)
            for ind in reg_ints:
                op_stack[ind] = max_level


            # +++ More edits to depth function +++
            if max(op_stack)> len(multi_qgate_layer):
                multi_qgate_layer.append(False)
            q2instr = False
            if instruction.operation.num_qubits > 1:
                q2instr = True
            if multi_qgate_layer[max_level-1] == False: # Does not get triggered if set to true so we dont overwrite
                multi_qgate_layer[max_level-1] = q2instr
            # +++ End of edits +++    


        # ++ Replace depth by time estimate ++
        for elem in multi_qgate_layer:
            if elem==True:
                time_estimate+=tg2
            else:
                time_estimate+=tg1
        return time_estimate


def prepare_base_qft(run_config):
    """
    Prepare a simple quantum circuit that prepares the same state as a Quantum Fourier Transform
    State must lay in between 0 and 2^n -1
    Caution: Does only work for states that are not in superposition (The base states) (See Qiskit textbook)
    Then qiskits general IQFT is appended. Purpose: Check outcome of noisy simulations
    """
    nqubits = run_config["nqubits"]
    state = run_config["winner_state"]


    circuit = QuantumCircuit(nqubits)
    [circuit.h(i) for i in range(nqubits)]
    
    for i in range(nqubits):
        phase =  ((2*np.pi*state)/(np.power(2,i+1)))#%(2*np.pi)
        circuit.p(phase,i)

    inv_QFT = QFT(num_qubits=nqubits,approximation_degree=0,inverse=True,do_swaps=False) # We use the QFT's internal swapping to counter Qiskits qubit ordering

    circuit.append(inv_QFT,circuit.qubits)

    return circuit


# +++ Helpers for the variational circuits +++
def encode_circ_paper_version(width: int = 8):
    """
    Implementation of the data encoder explained in
    Quantum circuit learning
    by Kosuke Mitarai, Makoto Negoro, Masahiro Kitagawa, Keisuke Fujii
    https://arxiv.org/abs/1905.10876

    Apply unitary as in Paper Quantum Circuit Learning; x \in [-1,1]
    """
    x = Parameter('x')
    encoder_circ = QuantumCircuit(width, 1)

    [encoder_circ.rz(np.arccos(x*x), q) for q in range(width)]
    [encoder_circ.ry(np.arcsin(x), q) for q in range(width)]

    return encoder_circ


def circ_elem_11(nqubits, L):
    """
    Expendable implementation of "circuit element 11" in
    Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms
    by Sukin Sim, Peter D. Johnson, Alan Aspuru-Guzik
    https://arxiv.org/abs/1905.10876

    L: repetitions
    This concept is only easily extentable for nqubuts%2 = 0
    """

    if (nqubits % 1) != 0:
        raise ValueError("This template only supports nqubits to be even")

    circ = QuantumCircuit(nqubits, 1)
    param_list = [Parameter("theta_" + str(i)) for i in range((4*nqubits-4)*L)]
    for layer in range(L):
        start_idx = (4*nqubits-4)*layer  # Start idx for Parameter list
        for i1 in range(nqubits):
            circ.ry(theta=param_list[start_idx+i1], qubit=i1)
        start_idx += nqubits
        for i2 in range(nqubits):
            circ.rz(phi=param_list[start_idx+i2], qubit=i2)
        start_idx += nqubits
        for i3 in np.arange(0, nqubits, 2):
            circ.cx(i3+1, i3)
        for i4 in range(nqubits-2):
            circ.ry(theta=param_list[start_idx+i4], qubit=i4+1)
        start_idx += (nqubits-2)
        for i5 in range(nqubits-2):
            circ.rz(phi=param_list[start_idx+i5], qubit=i5+1)
        start_idx += (nqubits-2)
        for i6 in np.arange(1, nqubits-1, 2):
            circ.cx(i6+1, i6)
    return circ


def target_fun(name:str = "pow(x,2)",count=50,targ_const=0.3):
    """
    Samples \in [-1,1] and returns the evaluated function (name) at that point
    In case name==constrx, the targ_const function will be returned in any case.

    In case for the same number of counts (~ training iteations), the permutation will
    be identical every time for fair comparison.
    """

    # Sample evenly distributed data fair every architecture
    lindat = np.linspace(-1,1,count)
    # Sample with seed so each run with same iterations will have same order of data points
    x = np.random.default_rng(111).permutation(lindat)

    match name:
        case "powx2":
            return (list(x),list(x*x))
        case "constrx":
            return ([None for _ in range(count)],[targ_const for _ in range(count)])
        
        case other:
            raise ValueError("Please add the requested function to target_fun")

def get_full_map(nqubits):
    """
    Returns full coupling list for nqubits
    """
    coupling = []
    for i in range(nqubits):
        for j in range(nqubits):
            if i != j:
                coupling.append((i, j))
    return coupling

def get_line_map(nqubits):
    """
    Returns a line map for nqubits
    """
    coupling = []
    for i in range(nqubits):
        coupling.append((i,i+1))

    return coupling

def get_qubits_for_coupling_map(coupling_map):
    return set(sum(coupling_map, []))

def get_nx_graph(base_map):
    qubits = get_qubits_for_coupling_map(base_map)
    g = nx.Graph()
    g.add_nodes_from(list(qubits))
    g.add_edges_from(base_map)
    return g

def get_eligible_edges(number_of_nodes, coupling_map, coupling_map_distances, distance):
    qubits = set(sum(coupling_map, []))
    eligible_edges = []
    for (i, j) in itertools.combinations(qubits, 2):
        if coupling_map_distances[i][j] == distance:
            eligible_edges.append((i, j))
    return eligible_edges

def increase_coupling_density(base_map, density):
    
    if density == 0:
        return base_map
    
    number_of_nodes = len(set(sum(base_map, [])))
    ## Note: IBM-Q requires a list of directed edges as a coupling map
    max_edges = (number_of_nodes)*(number_of_nodes-1)
    coupling_map = base_map.copy()
    
    g = get_nx_graph(coupling_map)
    coupling_map_distances = dict(nx.all_pairs_shortest_path_length(g))
    distance = 2 # Start with edges connecting nodes of distance 2
    eligible_edges = get_eligible_edges(number_of_nodes, base_map, coupling_map_distances, distance)
    
    while (len(coupling_map)/max_edges) < density:
        nodes = eligible_edges[np.random.choice(np.arange(len(eligible_edges)), size=1, replace=False)[0]]
        coupling_map.append([nodes[0], nodes[1]])
        coupling_map.append([nodes[1], nodes[0]])
        eligible_edges.remove(nodes)
        # Check if no more elements of the current minimum distance exist
        if not eligible_edges:
            # If so, increase the considered distance and fetch a new list of eligible edges
            distance = distance + 1
            eligible_edges = get_eligible_edges(number_of_nodes, base_map, coupling_map_distances, distance)
        
    return coupling_map
