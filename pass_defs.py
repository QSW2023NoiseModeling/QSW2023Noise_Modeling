# SPDX-License-Identifier: Apache-2.0

# (C) Copyright XXX 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# NOTE: In this file the following methods use derivations of source code from the Qiskit project.
# -> We based our passes on Qiskits "Transpiler Passes and Pass managers" tutorial

# We honor the previous contributions and publish own contributions 
# under the same Apache 2.0 License, to the best of our knowldedge, in compliance with the Apache-2.0 license terms.


from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library.standard_gates import *
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

from gate_defs import OwnGPI1Gate, OwnGPI2Gate, OwnXYGate, OwnX1Gate, OwnX3Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.extensions import UnitaryGate
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers import (
    level_0_pass_manager,
    level_1_pass_manager,
    level_2_pass_manager,
    level_3_pass_manager,
)
from qiskit.converters import dag_to_circuit
from qiskit.converters import circuit_to_dag
import numpy as np
from qiskit.transpiler import CouplingMap
from gate_defs import *
from copy import copy

# General NOTE:
# Here we add custom transpiler passes in order to
# 1) Transpile data on to the given gate sets
# 2) Transform the non-standard gates to Qsikit class "UnitaryGate" in order to make it compatible with the Aer backend

class RigettiTranslator(TransformationPass):
    """
    In order to transpile to Rigettis gate-set using Qiskits compiler,
    we need to add equivalences of Qiskit supported gates and compositions of gates
    that involve our own definitions.
    Some of the equivalences were found with help of the paper:
    Quantum Circuit Identities,
    by Chris Lomont,
    https://arxiv.org/abs/quant-ph/0307111
    """

    def __init__(self,run_config,hw_config):
        self._run_config = run_config
        self._hw_config = hw_config
        super().__init__()

    def run(self,dag):
        """
        Run pass
        """

        # Add Rigetti X1 Gate to equivalence library
        # S => H OwnX1 H 
        q = QuantumRegister(1,"q")
        def_x1 = QuantumCircuit(q)
        def_x1.append(HGate(),[q[0]],[])
        def_x1.append(OwnX1Gate(),[q[0]],[])
        def_x1.append(HGate(),[q[0]],[])
        sel.add_equivalence(SGate(), def_x1)

        
        # Sdg => H OwnX3 H
        q = QuantumRegister(1,"q")
        def_x3 = QuantumCircuit(q)
        def_x3.append(HGate(),[q[0]],[])
        def_x3.append(OwnX3Gate(),[q[0]],[])
        def_x3.append(HGate(),[q[0]],[])
        sel.add_equivalence(SdgGate(), def_x3)

        # H =>  S X1 S
        q = QuantumRegister(1,"q")
        def_h1 = QuantumCircuit(q)
        def_h1.append(SGate(),[q[0]],[])
        def_h1.append(OwnX1Gate(),[q[0]],[])
        def_h1.append(SGate(),[q[0]],[])
        sel.add_equivalence(HGate(), def_h1)

        # H => Z1=S H X3
        q = QuantumRegister(1,"q")
        def_h2 = QuantumCircuit(q)
        def_h2.append(SGate(),[q[0]],[])
        def_h2.append(HGate(),[q[0]],[])
        def_h2.append(OwnX3Gate(),[q[0]],[])
        sel.add_equivalence(HGate(), def_h2)

        # H => X1 H sdg
        q = QuantumRegister(1,"q")
        def_h3 = QuantumCircuit(q)
        def_h3.append(OwnX1Gate(),[q[0]],[])
        def_h3.append(HGate(),[q[0]],[])
        def_h3.append(SdgGate(),[q[0]],[])
        sel.add_equivalence(HGate(), def_h3)

        # Y => X3 Z X1
        q = QuantumRegister(1,"q")
        def_y1 = QuantumCircuit(q)
        def_y1.append(OwnX3Gate(),[q[0]],[])
        def_y1.append(ZGate(),[q[0]],[])
        def_y1.append(OwnX1Gate(),[q[0]],[])
        sel.add_equivalence(YGate(), def_y1)

        gates = ['x','OwnX1','OwnX3','rz','cz','cp','xx_plus_yy'] 
        level = self._run_config["opt_level"]
        pm_config = PassManagerConfig(basis_gates=gates,coupling_map=CouplingMap(self._hw_config["coupling_map"]))
        match level:
            case 0:
                pm = level_0_pass_manager(pm_config)
            case 1:
                pm = level_1_pass_manager(pm_config)
            case 2:
                pm = level_2_pass_manager(pm_config)
            case 3:
                pm = level_3_pass_manager(pm_config)

        dag = pm.run(dag)
        # Copy layout for later reordering of the bits, since we loose that in the unitarizer step
        the_layout = copy(dag._layout)

        to_unitary = Unitarizer(gates=["OwnX1","OwnX3"])
        dag = to_unitary.run(circuit_to_dag(dag))

        circo = dag_to_circuit(dag)
        circo._layout = the_layout
        return circo


class IonQ_Translator(TransformationPass):
    """
    Call this to transpile from rx,ry,rz, 
    to GPi1, GPi2, RXX :
    Based in informatipn provided in:
    https://ionq.com/docs/getting-started-with-native-gates
    and
    D. Maslov, Basic circuit compilation techniques for an ion-trap quantum machine

    Step 1: Map to coupling map and rx, ry, rz, cx; (Using Qiskit pass, with chosen optimization level)
    Step 2: Map CNOTs to rxx, rx and ry
    Step 3: Map rx, ry, rz to gpi1 and gpi2 (First: Check whether rx, ry have angles close to pi/2, for minor optimization!)
    (Step 4: Replace rxx with MS gate of theta = pi/4; We ommit this step, since in our noise model they have identical noise level)
    Step 5: Replace gates with UnitaryGate s.t. they may be simulated
    """

    def __init__(self):
        """
        NOTE: IONQ does have full coupling, so no need to compile to other coupling graphs.
        """
        super().__init__()

    def run(self, dag,run_config):
        """ 
        Run pass
        """     

        # +++ Step 1 +++
        level = run_config["opt_level"]
        interim_gates = ['rx', 'ry', 'rz', 'cx']  #


        pm_config = PassManagerConfig(basis_gates=interim_gates)
        match level:
            case 0:
                pm = level_0_pass_manager(pm_config)
            case 1:
                pm = level_1_pass_manager(pm_config)
            case 2:
                pm = level_2_pass_manager(pm_config)
            case 3:
                pm = level_3_pass_manager(pm_config)

        circo = pm.run(dag)
        the_layout = copy(circo._layout)
        dag = circuit_to_dag(circo)

        # +++ Step 2 +++
        while dag.op_nodes(op=CXGate):
            old_node = dag.op_nodes(op=CXGate).pop()
            q = old_node.qargs
            sub_dag = DAGCircuit()
            sub_dag.add_qubits(q)
        
            sub_dag.apply_operation_back(RYGate(theta=np.pi/2),qargs=[q[0]])
            sub_dag.apply_operation_back(RXXGate(theta=np.pi/2),qargs=[q[0], q[1]]) #  Here it does not matter since rxx is symmetrical
            sub_dag.apply_operation_back(RXGate(theta=-np.pi/2),qargs=[q[0]])
            sub_dag.apply_operation_back(RXGate(theta=-np.pi/2),qargs=[q[1]])
            sub_dag.apply_operation_back(RYGate(theta=-np.pi/2),qargs=[q[0]])

            dag.substitute_node_with_dag(node = old_node, input_dag=sub_dag,wires=[q[0],q[1]])

        # +++ Step 3 (&4) +++
        while dag.op_nodes(op=RXGate):
            sub_dag = DAGCircuit()
            r = QuantumRegister(1)
            sub_dag.add_qreg(r)

            old_node = dag.op_nodes(op=RXGate).pop()
            # NOTE: I cast here since sometimes; the dataType is ParameterExpression but with a floating point value.
            theta = float(old_node.op.params[0]) 

            if np.abs(theta-np.pi/2)< 1e-6:
                sub_dag.apply_operation_back(OwnGPI2Gate(phi=0),qargs=[r[0]])
            elif np.abs(theta+np.pi/2)<1e-6:
                sub_dag.apply_operation_back(OwnGPI2Gate(phi=np.pi),qargs=[r[0]])
            elif np.abs(theta-np.pi)<1e-6:
                sub_dag.apply_operation_back(OwnGPI1Gate(phi=0),qargs=[r[0]])
            elif np.abs(theta+np.pi)<1e-6:
                sub_dag.apply_operation_back(OwnGPI1Gate(phi=np.pi),qargs=[r[0]])
            else:
                sub_dag.apply_operation_back(OwnGPI2Gate(phi=1.5*np.pi),qargs=[r[0]])
                sub_dag.apply_operation_back(RZGate(phi=theta),qargs=[r[0]])       
                sub_dag.apply_operation_back(OwnGPI2Gate(phi=0.5*np.pi),qargs=[r[0]])
            dag.substitute_node_with_dag(node = old_node, input_dag=sub_dag,wires=[r[0]])

        while dag.op_nodes(op=RYGate):
            sub_dag = DAGCircuit()
            r = QuantumRegister(1)
            sub_dag.add_qreg(r)

            old_node = dag.op_nodes(op=RYGate).pop()
            
            theta = float(old_node.op.params[0])
            if np.abs(theta-np.pi/2)< 1e-6:
                sub_dag.apply_operation_back(OwnGPI2Gate(phi=np.pi*0.5),qargs=[r[0]])
            elif np.abs(theta+np.pi/2)<1e-6:
                sub_dag.apply_operation_back(OwnGPI2Gate(phi=1.5*np.pi),qargs=[r[0]])
            elif np.abs(theta-np.pi)<1e-6:
                sub_dag.apply_operation_back(OwnGPI1Gate(phi=np.pi*0.5),qargs=[r[0]])
            elif np.abs(theta+np.pi)<1e-6:
                sub_dag.apply_operation_back(OwnGPI1Gate(phi=1.5*np.pi),qargs=[r[0]])
            else:
                sub_dag.apply_operation_back(OwnGPI2Gate(phi=0),qargs=[r[0]])
                sub_dag.apply_operation_back(RZGate(phi=theta),qargs=[r[0]])
                sub_dag.apply_operation_back(OwnGPI2Gate(phi=np.pi),qargs=[r[0]])

            dag.substitute_node_with_dag(node = old_node, input_dag=sub_dag,wires=[r[0]])


        # +++ Step 5 +++
        to_unitary = Unitarizer(gates=["OwnGPI1","OwnGPI2"])
        dag = to_unitary.run(dag)

        circo = dag_to_circuit(dag)
        circo._layout = the_layout
        return circo

class Unitarizer(TransformationPass):
    """
    It appears that Qiskit's Aer backend does not support unknown gate definitions for simulation,
    except the case they have the data type 'UnitaryGate' (of course all custom gates are unitary anyway)
    --> After transpilation we change all custom gates to the UnitaryGate type.
    (NOTE: We could not do this from the very beginning, since the transpile step to custom gates would not have worked.)
    """

    def __init__(self, gates):
        super().__init__()
        self._gate_list = gates
    def run(self, dag):
        """
        Run Unitarizer
        """
        for gate_name in self._gate_list:
            if gate_name.startswith('Own'):
                match gate_name:
                    case 'OwnGPI1':
                        while dag.op_nodes(op=OwnGPI1Gate):
                            sub_dag = DAGCircuit()
                            r = QuantumRegister(1, 'r')
                            old_node = dag.op_nodes(op=OwnGPI1Gate).pop()
                            u_gate = UnitaryGate(
                                old_node.op.to_matrix(), label=gate_name)
                            sub_dag.add_qreg(r)
                            sub_dag.apply_operation_back(u_gate, [r[0]])
                            dag.substitute_node_with_dag(
                                node=old_node, input_dag=sub_dag, wires=[r[0]])
                    case 'OwnGPI2':
                        while dag.op_nodes(op=OwnGPI2Gate):
                            sub_dag = DAGCircuit()
                            r = QuantumRegister(1,'r')
                            old_node = dag.op_nodes(op=OwnGPI2Gate).pop()
                            u_gate = UnitaryGate(
                                old_node.op.to_matrix(), label=gate_name)
                            sub_dag.add_qreg(r)
                            sub_dag.apply_operation_back(u_gate, [r[0]])
                            dag.substitute_node_with_dag(
                                node=old_node, input_dag=sub_dag, wires=[r[0]])
                            
                    case 'OwnX1':
                        while dag.op_nodes(op=OwnX1Gate):
                            sub_dag = DAGCircuit()
                            r = QuantumRegister(1,'r')
                            old_node = dag.op_nodes(op=OwnX1Gate).pop()
                            u_gate = UnitaryGate(
                                old_node.op.to_matrix(), label=gate_name)
                            sub_dag.add_qreg(r)
                            sub_dag.apply_operation_back(u_gate, [r[0]])
                            dag.substitute_node_with_dag(
                                node=old_node, input_dag=sub_dag, wires=[r[0]])
                            
                    case 'OwnX3':
                        while dag.op_nodes(op=OwnX3Gate):
                            sub_dag = DAGCircuit()
                            r = QuantumRegister(1,'r')
                            old_node = dag.op_nodes(op=OwnX3Gate).pop()
                            u_gate = UnitaryGate(
                                old_node.op.to_matrix(), label=gate_name)
                            sub_dag.add_qreg(r)
                            sub_dag.apply_operation_back(u_gate, [r[0]])
                            dag.substitute_node_with_dag(
                                node=old_node, input_dag=sub_dag, wires=[r[0]])

                    case 'OwnXY':
                        while dag.op_nodes(op=OwnXYGate):
                            sub_dag = DAGCircuit()
                            r = QuantumRegister(2, 'r')
                            old_node = dag.op_nodes(op=OwnXYGate).pop()
                            u_gate = UnitaryGate(
                                old_node.op.to_matrix(), label=gate_name)
                            sub_dag.add_qreg(r)
                            sub_dag.apply_operation_back(u_gate, [r[0], r[1]])
                            dag.substitute_node_with_dag(
                                node=old_node, input_dag=sub_dag, wires=[r[0], r[1]])
                    case _:
                        print(
                            "Sorry, please add your non-standard gate to pass_defs.py first!")
                        return

            else:  # Only transpile non-standard gates to unitary
                continue

        return dag

