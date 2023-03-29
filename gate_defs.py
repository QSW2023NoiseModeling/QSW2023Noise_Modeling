# SPDX-License-Identifier: Apache-2.0

# (C) Copyright IBM 2017, 2019

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# (C) Copyright XXX 2023.

# NOTE: In this file the following methods use derivations of source code from the Qiskit project.
# 1) OwnGPI1, OwnGPI2: Change only parameters of qiskit/circuit/library/r.py (RGate)
# 2) OwnX1, OwnX3: Change only parameters of qiskit/circuit/library/rx.py (RXGate)
# 3) OwnX1, OwnX3: Change only parameters of qiskit/circuit/library/rx.py (RXGate)

# We honor the previous contributions and publish own contributions 
# under the same Apache 2.0 License, to the best of our knowldedge, in compliance with the Apache-2.0 license terms.


import math
from cmath import exp
from typing import Optional, Union
import numpy as np
from qiskit.qasm import pi
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit import QuantumCircuit

from qiskit.circuit.library.standard_gates import RZGate,SXGate,SGate,CXGate,RYGate,SdgGate,SXdgGate, RGate, CRXGate, RXGate


# +++ NOTE: From here: IONQ gatesets +++
# 1) GPI1 (custom R Gate with theta=pi)
# 2) GPI2 (custom R Gate with theta = pi/2)
# 3) RZ Gate --> native
# 4) MS Gate : Use cx equivalence such that the rxx gate equals ms gate at specific angle.
# See https://ionq.com/docs/getting-started-with-native-gates
class OwnGPI1Gate(Gate):
    """
    Create GPI1 Gate from R gate codebase theta = pi; then up to a global phase we have the same gate!
    """
    def __init__(
        self, phi: ParameterValueType, label: Optional[str] = None
    ):
        super().__init__("OwnGPI1", 1, [phi], label=label)

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from qiskit.circuit.library.standard_gates import U3Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        theta = np.pi
        phi = self.params[0]
        rules = [(U3Gate(theta, phi - pi / 2, -phi + pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """
        NOTE: I just set theta to pi so we can ommit this
        """
        return OwnGPI1Gate(self.params[0])


    def __array__(self, dtype=None):
        theta, phi = np.pi, float(self.params[0])
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        exp_m = exp(-1j * phi)
        exp_p = exp(1j * phi)
        return np.array([[cos, -1j * exp_m * sin], [-1j * exp_p * sin, cos]], dtype=dtype)

    def power(self, exponent: float):
        theta = np.pi 
        phi = self.params[0]
        return OwnGPI1Gate(exponent * theta, phi)
    


class OwnGPI2Gate(Gate):
    """
    Create GPI1 Gate from R Gate codebase theta = pi/2; then up to a global phase we have the same gate!
    """
    def __init__(
        self, phi: ParameterValueType, label: Optional[str] = None
    ):
        super().__init__("OwnGPI2", 1, [phi], label=label)

    def _define(self):

        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from qiskit.circuit.library.standard_gates import U3Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        theta = np.pi/2
        phi = self.params[0]
        rules = [(U3Gate(theta, phi - pi / 2, -phi + pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """
        NOTE: I just set theta to pi so we can ommit this
        """
        return OwnGPI1Gate(self.params[0])


    def __array__(self, dtype=None):
        theta, phi = np.pi/2, float(self.params[0])
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        exp_m = exp(-1j * phi)
        exp_p = exp(1j * phi)
        return np.array([[cos, -1j * exp_m * sin], [-1j * exp_p * sin, cos]], dtype=dtype)

    def power(self, exponent: float):
        theta = np.pi/2 
        phi = self.params[0]
        return OwnGPI1Gate(exponent * theta, phi)
    


# +++ NOTE: From here: Rigetti Gates +++

# 1) owmXy: xx_plus_yy with beta=0 NOTE: We added the parametrized xx_plus_yy gate to the rigetti gateset, but it gets never compiled to.
# 2) CPhase is qiskit-native cp gate
# 3) cz is also qiskit native
# 4) rx needs to be custom
# 5) rz native
# --> gate Set Rigetti complete

class OwnXYGate(Gate):
    """
    Create XY gate from XXplusYY Gate with beta = 0
    """

    def __init__(
        self,
        theta: ParameterValueType,
        label: Optional[str] = "OwnXY",
    ):

        super().__init__("xx_plus_yy", 2, [theta], label=label)

    def _define(self):
        theta = self.params[0]
        beta = 0
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (RZGate(beta), [q[0]], []),
            (RZGate(-pi / 2), [q[1]], []),
            (SXGate(), [q[1]], []),
            (RZGate(pi / 2), [q[1]], []),
            (SGate(), [q[0]], []),
            (CXGate(), [q[1], q[0]], []),
            (RYGate(-theta / 2), [q[1]], []),
            (RYGate(-theta / 2), [q[0]], []),
            (CXGate(), [q[1], q[0]], []),
            (SdgGate(), [q[0]], []),
            (RZGate(-pi / 2), [q[1]], []),
            (SXdgGate(), [q[1]], []),
            (RZGate(pi / 2), [q[1]], []),
            (RZGate(-beta), [q[0]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        return OwnXYGate(-self.params[0])
    
    def __array__(self, dtype=complex):
        half_theta = float(self.params[0]) / 2
        beta = 0
        cos = math.cos(half_theta)
        sin = math.sin(half_theta)
        return np.array(
            [
                [1, 0, 0, 0],
                [0, cos, -1j * sin * exp(-1j * beta), 0],
                [0, -1j * sin * exp(1j * beta), cos, 0],
                [0, 0, 0, 1],
            ],
            dtype=dtype,
        )

    def power(self, exponent: float):
        theta =  self.params[0]
        return OwnXYGate(exponent * theta)


class OwnX1Gate(Gate):
    """
    X1 Gate; meaning Rx(+pi/2) (up to global phase difference)
    """

    def __init__(self, label: Optional[str] = None):
        super().__init__("OwnX1", 1, [], label=label)

    def _define(self):
        """
        Nix
        """
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RGate(np.pi/2, 0), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """
        Nix
        """
        if num_ctrl_qubits == 1:
            gate = CRXGate(np.pi/2, label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)


    def inverse(self):
        r"""Return inverted RX gate.

        :math:`RX(\lambda)^{\dagger} = RX(-\lambda)`
        """
        return RXGate(-np.pi/2)


    def __array__(self, dtype=None):
        """Return a numpy.array for the RX gate."""
        cos = math.cos((np.pi/2) / 2)
        sin = math.sin((np.pi/2) / 2)
        return np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=dtype)

    def power(self, exponent: float):
        """Raise gate to a power."""
        return RXGate(exponent * (np.pi/2))
    

class OwnX3Gate(Gate):
    """
    X3 Gate; meaning Rx(-pi/2) (up to global phase difference)
    """

    def __init__(self, label: Optional[str] = None):
        """Create new RX gate."""
        super().__init__("OwnX3", 1, [], label=label)

    def _define(self):
        """
        Nix
        """
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RGate(-np.pi/2, 0), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """
        Nix
        """
        if num_ctrl_qubits == 1:
            gate = CRXGate(-np.pi/2, label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)


    def inverse(self):
        r"""Return inverted RX gate.

        :math:`RX(\lambda)^{\dagger} = RX(-\lambda)`
        """
        return RXGate(np.pi/2)


    def __array__(self, dtype=None):
        """Return a numpy.array for the RX gate."""
        cos = math.cos((-np.pi/2) / 2)
        sin = math.sin((-np.pi/2) / 2)
        return np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=dtype)

    def power(self, exponent: float):
        """Raise gate to a power."""
        return RXGate(exponent * (-np.pi/2))
