# SPDX-License-Identifier: Apache-2.0

# (C) Copyright IBM 2018, 2019.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# (C) Copyright XXX 2023.



# NOTE: In this file the following methods use derivations of source code from the Qiskit project.
# 1) hw_to_noise
# - Based on Qiskits basic_device_gate_errors and _device_depolarizing_error (See. qiskit_aer/noise/device/models.py)
# - We use the construction of thermal relaxation error and depolarizing error as given in the above files,
# - however simplify it such that only average data for qubits 
# 2) truncate_t2_value:
# - This helper function was reused from qiskit_aer/noise/device/models.py 

# We honor the previous contributions and publish own contributions 
# under the same Apache 2.0 License, to the best of our knowldedge, in compliance with the Apache-2.0 license terms.

import os
import copy
from functools import partial
import numpy as np
import pandas as pd
from qiskit.quantum_info import average_gate_fidelity, DensityMatrix, process_fidelity, state_fidelity, partial_trace
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error
from qiskit_aer.backends import AerSimulator
from qiskit_aer.library.save_instructions import SaveDensityMatrix
from qiskit import transpile, ClassicalRegister
from qiskit.circuit.library.standard_gates import *
import datetime
import multiprocessing
from multiprocessing import get_context
import argparse
# Own libs
from gate_defs import *
from pass_defs import IonQ_Translator, RigettiTranslator
from helper_functions import *
from grover_lib import *


def parse_args():
    """
    Parse input arguments | Subparser for every algorithm with own default values
    Returns run_config
    """
    parser = argparse.ArgumentParser(
        prog='RunSimulation', description='Run quantum simulations')
    subparser = parser.add_subparsers(dest='alg')

    # +++ Common arguments +++
    parser.add_argument('--hw', type=str, default="ibmq_kolkata",
                        help="Options: ibmq_kolkata | ionq_aria | rigetti_aspenm3 or your own custom configs")
    parser.add_argument('--serial_reps', type=int, default=1,
                        help="Serial Repetitions of simulation)")

    parser.add_argument('--noise_method', type=str, default='inherent',
                        help="inherent: From fidelities; Other Options : inherent | depol | x | y | z ")
    parser.add_argument("--noise_level", type=float, default=-1,
                        help="Valid options [0,1); Ignored otherwise; ignored for inherent option")

    parser.add_argument('--method', type=str,
                        default='density_matrix', help="Aer Simulation method (Currently only supported option)")
    parser.add_argument('--opt_level', type=int,
                        default=1, help="Optimization level for qiskit compiler that may be used in the transpile procedure")

    # +++ Grover arguments +++
    grover = subparser.add_parser('grover')
    grover.add_argument('--nqubits', type=int, nargs=2, default=[2, 3],
                        help="Simulates for all nqubits between first and second arg (inclusive) | Ifarg1==arg2: Only one simulation starts")
    grover.add_argument("--nbshots", type=int, default=0,
                        help='Number of runs through circuit per experiment (Currently, for density sim: Set to zero)')
    grover.add_argument('--par', action='store_true',
                        help="Starts parallel sim for range(nubits[1],nqubits[2]+1) | Oherwise serial execution starts")
    grover.add_argument('--winner_state', type=int,
                        default=1, help="Marked element in oracle")
    grover.add_argument('--parallel_reps', type=int, default=1,
                        help="Parallel Repetitions of all simulations (Can be quite RAM and CPU hungry)")

    # +++ IQFT Arguments +++
    iqft = subparser.add_parser('iqft')
    iqft.add_argument('--nqubits', type=int, nargs=2, default=[2, 3],
                      help=" Simulates for all nqubits between first and second arg (inclusive) | Ifarg1==arg2: Only one simulations starts")
    iqft.add_argument("--nbshots", type=int, default=0,
                      help='Number of runs through circuit per experiment (Currently, for density sim: Set to zero)')
    iqft.add_argument('--par', action='store_true',
                      help="Starts parallel sim for range(nubits[1],nqubits[2]+1) | Oherwise serial execution starts")
    iqft.add_argument('--winner_state', type=int,
                      default=1, help="Target state iqft")
    iqft.add_argument('--parallel_reps', type=int, default=1,
                      help="Parallel Repetitions of all simulations (Can be quite RAM and CPU hungry)")

    # Parser for variational circuit
    var = subparser.add_parser('var')
    var.add_argument('--eta', type=float, default=0.1,
                     help="Step Size in gradient descent")
    var.add_argument('--batch_size', type=int, default=1,
                     help="Batch size for training (Currently only 1 is supported)")
    var.add_argument('--nbshots', type=int, default=0,
                     help="Number of runs through circuit per experiment (Currently, for density sim: Set to zero)")
    var.add_argument('--iters', type=int, default=50,
                     help="Number of training iterations")
    var.add_argument('--L', type=int, default=1,
                     help="Number of repetitions of circuit element")
    var.add_argument('--nqubits', type=int, default=4,
                     help='Number of qubits (Width of circuit)')
    var.add_argument('--par', action='store_true',
                     help="Leverage parallel gradient computation")
    var.add_argument('--fun', type=str, default='powx2',
                     help="Specify to-be approximated function")
    var.add_argument('--measure_pos',type=int,default=1,help="Position of measurement for one-d variational approximation problems (idx starts from zero)")

    args = parser.parse_args()

    if args.alg == 'grover':
        return parse_grover(args)
    elif args.alg == 'iqft':
        return parse_iqft(args)
    elif args.alg == 'var':
        return parse_var(args)
    else:  # Debug Mode
        pass
        # raise ValueError("Invalid choice of algorithm")


def parse_grover(args):
    """
    Returns config_list, args.par
    """
    config_list = []
    if args.par:
        inner_list = [{
            "algo": "grover",
                    "nqubits": q,
                    "hw": args.hw,
                    "method": args.method,
                    "opt_level": args.opt_level,
                    "nbshots": args.nbshots,
                    "winner_state": args.winner_state,
                    "noise_method": args.noise_method,
                    "noise_level": args.noise_level,
        } for q in range(args.nqubits[0], args.nqubits[1]+1)]

        inner_list = inner_list * args.parallel_reps
        config_list.append(inner_list)
        config_list = config_list * args.serial_reps
    else:
        for q in range(args.nqubits[0], args.nqubits[1]+1):
            config_list.append([{
                "algo": "grover",
                "nqubits": q,
                "hw": args.hw,
                "method": args.method,
                "opt_level": args.opt_level,
                "nbshots": args.nbshots,
                "winner_state": args.winner_state,
                "noise_method": args.noise_method,
                "noise_level": args.noise_level,
            }])
            config_list = config_list * args.serial_reps
    return config_list


def parse_iqft(args):
    """
    config_list, args.par
    """
    config_list = []
    if args.par:
        inner_list = [{
            "algo": "iqft",
            "nqubits": q,
            "hw": args.hw,
            "method": args.method,
            "opt_level": args.opt_level,
            "nbshots": args.nbshots,
            "winner_state": args.winner_state,
            "noise_method": args.noise_method,
            "noise_level": args.noise_level,
        } for q in range(args.nqubits[0], args.nqubits[1]+1)]

        inner_list = inner_list * args.parallel_reps
        config_list.append(inner_list)
        config_list = config_list * args.serial_reps
    else:
        for q in range(args.nqubits[0], args.nqubits[1]+1):
            config_list.append([{
                "algo": "grover",
                "nqubits": q,
                "hw": args.hw,
                "method": args.method,
                "opt_level": args.opt_level,
                "nbshots": args.nbshots,
                "winner_state": args.winner_state,
                "noise_method": args.noise_method,
                "noise_level": args.noise_level,
            }])
            config_list = config_list * args.serial_reps
    return config_list


def parse_var(args):
    """
    config_list, args.par
    """
    config_list = [[{"algo": "var",
                     "hw": args.hw,
                     "fun": args.fun,
                     "eta": args.eta,
                     "nqubits": args.nqubits,
                     "nbshots": args.nbshots,
                     "batch_size": args.batch_size,
                     "iterations": args.iters,
                     "L": args.L,
                     "method": args.method,
                     "opt_level": args.opt_level,
                     "noise_level": args.noise_level,
                     "noise_method": args.noise_method,
                     "par": args.par,
                     "measure_pos":args.measure_pos
                     }]*args.serial_reps]
    return config_list


def main(config_list):

    # +++ This config is only used if run_simulation starts in an ide.
    #  This enables debugging and quick testing. Else the CLI is suggested +++
    if config_list == None:
        config_list = [[{"algo": "var",
                         "hw": "ibmq_kolkata",
                         "fun": "powx2",
                         "eta": 1.0,
                         "nqubits": 4,
                         "nbshots": 0,
                         "batch_size": 1,
                         "iterations": 5,
                         "L": 1,
                         "method": "density_matrix",
                         "noise_method": "x", 
                         "opt_level": 1,
                         "noise_level": 0.001,
                         "measure_pos": 0,
                         "par": True
                         }]*1]
    # +++ End of config definition+++

    # +++ Config template for Grover / IQFT algorithm (change params) +++
    # if config_list == None:
    #     config_list = []
    #     inner_list = [{
    #         "algo": "grover",
    #                 "nqubits": q,
    #                 "hw": "ibmq_kolkata",
    #                 "method": "density_matrix",  # density_matrix
    #                 "opt_level": 1,
    #                 "nbshots": 0,
    #                 "winner_state": 1,
    #                 "noise_method": 'inherent',
    #                 "noise_level": 0.001,
    #     } for q in range(2, 3)]

    #     inner_list = inner_list * 7  # Par reps
    #     config_list.append(inner_list)
    #     config_list = config_list * 1  # Serial reps
    # +++ End Debugging +++


    for iter, exec_unit in enumerate(config_list):
        print(
            f"+++ Start {len(exec_unit)} parallel calculations. Serial {iter+1} of {len(config_list)} +++")
        if len(exec_unit) == 1:  # Start Serial execution
            run_config = exec_unit[0]
            hw_config = np.load(
                "./backend_data/"+run_config["hw"] + ".npy", allow_pickle=True).item()

            if run_config["algo"] == "var":
                run_var(hw_config, run_config)
            else:
                run(run_config=run_config, hw_config=hw_config)
        else:  # Execute all configs in exec unit in parallel
            # Create partial function with hw_config preset for all entries
            hw_config = np.load(
                "./backend_data/"+exec_unit[0]["hw"] + ".npy", allow_pickle=True).item()
            par_run = partial(run, hw_config=hw_config)

            with get_context("spawn").Pool(processes=min(len(exec_unit), multiprocessing.cpu_count())) as pool:
                pool.map(par_run, exec_unit)
                pool.close()

    print("\nFinished all runs successfully!")


def run_var(hw_config: dict, run_config: dict):
    """
    Runtime Routine for variational circuits
    """

    print("Simulating {} with {}".format(
        hw_config["hw_name"], str(run_config.items())))
    # Unpack some parameters
    nqubits = run_config["nqubits"]
    batch_size = run_config["batch_size"]
    eta = run_config["eta"]
    s = run_config["nbshots"]


    # Set noise model
    match run_config["noise_method"]:
        case "inherent":
            noise_model = hw_to_noise(hw_config)
        case "depol":
            noise_model = pauli_depol(
                hw_config=hw_config, run_config=run_config)
        case "x":
            noise_model = pauli_x(hw_config=hw_config, run_config=run_config)
        case "y":
            noise_model = pauli_y(hw_config=hw_config, run_config=run_config)
        case "z":
            noise_model = pauli_z(hw_config=hw_config, run_config=run_config)
        case "none":
            noise_model = NoiseModel()
        case other:
            raise ValueError("Please choose valid options for noise_method")

    backend = AerSimulator(noise_model=noise_model,
                           method=run_config["method"], coupling_map=hw_config["coupling_map"])

    match run_config["fun"]:
        case "powx2":
            num_params = (4*nqubits-4)*run_config["L"]
            var_circ_param = circ_elem_11(run_config["nqubits"], run_config["L"])
            enc_circ_param = encode_circ_paper_version(nqubits)
            params = np.zeros(num_params)  # Initial param values
        case "constrx":
            num_params = 1    
            var_circ_param = QuantumCircuit(1)
            theta = Parameter('theta')
            var_circ_param.rx(theta, 0)
            enc_circ_param = None
            params = np.ones(num_params)*2*np.pi-0.1 # Such that we are not stuck ;D
        case other:
            raise ValueError("Invalid Choice of variational function")

    loss_saver_noisy = []
    loss_saver_ideal = []
    fidelity_saver = []
    param_saver = []
    xi_saver = []
    fxi_saver = []
    expec_saver_noisy = [] # TODO: Only works for batch sizes == 1!
    expec_saver_ideal = [] # TODO: Only works for batch sizes == 1!
    noisy_dens_saver = []
    ideal_dens_saver = []


    xdat, fxdat = target_fun(
        run_config["fun"], run_config["iterations"]*batch_size)
    for train_loop in range(run_config["iterations"]):

        batch_gradient = np.zeros_like(params)
        loss = 0
        loss_noisy = 0
        loss_ideal = 0
        for batch_loop in range(batch_size):

            xi = xdat.pop()
            fxi = fxdat.pop()

            if enc_circ_param is not None: # Case if input exists
                enc_circ_set = enc_circ_param.bind_parameters([xi])
                var_circ_set = var_circ_param.bind_parameters(params)
                t_circ = enc_circ_set.compose(var_circ_set)
                dm_ideal = DensityMatrix(t_circ)
                t_circ = custom_transpiler(t_circ, backend, hw_config, run_config)
            else: # Simple case without input
                var_circ_set = var_circ_param.bind_parameters(params)
                enc_circ_set = None
                t_circ = var_circ_set
                dm_ideal = DensityMatrix(t_circ)
                       
            order = qeorder(t_circ._layout,run_config["nqubits"])

            # Attach density matrix instruction in correct order
            dmatrix_noisy_instr = SaveDensityMatrix(run_config["nqubits"])
            t_circ.append(dmatrix_noisy_instr, order)

            result = backend.run(t_circ, shots=s).result()
            dm_noisy = result.data()["density_matrix"]

            if run_config["nqubits"] > 1:
                dm_red_ideal = partial_trace(dm_ideal,[i for i in range(run_config["measure_pos"])] + [j for j in range(run_config["measure_pos"]+1,run_config["nqubits"])])
                dm_red_noisy = partial_trace(dm_noisy,[i for i in range(run_config["measure_pos"])] + [j for j in range(run_config["measure_pos"]+1,run_config["nqubits"])])
            else:
                dm_red_ideal = dm_ideal
                dm_red_noisy = dm_noisy


            probs_noisy = dm_red_noisy.probabilities_dict()
            probs_ideal = dm_red_ideal.probabilities_dict() # Save to later see whether the noisy free variant would have produced better - or worse results

            state_fid = state_fidelity(dm_noisy,dm_ideal)
            fidelity_saver.append(state_fid)

            if len(probs_noisy.keys())==2:
                expec_theta_noisy = (probs_noisy['0']-probs_noisy['1'])
            elif '1' in probs_noisy.keys():
                expec_theta_noisy = -1
            else:
                expec_theta_noisy = 1

            if len(probs_ideal.keys())==2:
                expec_theta_ideal = (probs_ideal['0']-probs_ideal['1'])
            elif '1' in probs_ideal.keys():
                expec_theta_ideal = -1
            else:
                expec_theta_ideal = 1

            loss_noisy += 0.5*(expec_theta_noisy-fxi)**2  # Choose quadratic loss
            loss_ideal += 0.5*(expec_theta_ideal-fxi)**2  # Choose quadratic loss


            if num_params > 1 and run_config["par"]==True:
                # Calculate sub-gradient for every gradient in parallel
                partial_grad_fun = partial(gradient_calc, var_circ_param=var_circ_param, enc_circ_set=enc_circ_set,
                                        params=params, backend=backend, hw_config=hw_config, run_config=run_config)
                jiter = [j for j in range(num_params)]
                with get_context("spawn").Pool(processes=min(num_params, multiprocessing.cpu_count())) as pool:
                    expec_res = pool.map(partial_grad_fun, jiter)
                    pool.close()
                expec_plus, expec_minus = zip(*expec_res)

            elif num_params > 1 and run_config["par"]==False:
                partial_grad_fun = partial(gradient_calc, var_circ_param=var_circ_param, enc_circ_set=enc_circ_set,params=params, backend=backend, hw_config=hw_config, run_config=run_config)
                jiter = [j for j in range(num_params)]
                with get_context("spawn").Pool(processes=1) as pool:
                    expec_res = pool.map(partial_grad_fun, jiter)
                    pool.close()
                expec_plus, expec_minus = zip(*expec_res)

            else:
                expec_plus, expec_minus  = gradient_calc(0,var_circ_param=var_circ_param, enc_circ_set=enc_circ_set,params=params, backend=backend, hw_config=hw_config, run_config=run_config)
                expec_plus = [expec_plus]
                expec_minus = [expec_minus]

            gradient = np.array(
                [(expec_theta_noisy-fxi)*(expec_plus[j]-expec_minus[j])/2 for j in range(num_params)])
            batch_gradient += gradient

        # Here we have processed on batch of input data and accumulated the gradients! Its time to update the parameters
        batch_gradient /= batch_size
        params = (params - eta*batch_gradient) % (2*np.pi) 
        if train_loop % 1 == 0:
            print(f"Loss in iteration {train_loop} = {loss_noisy/batch_size}")
        # loss_saver.append((train_loop, loss_noisy/batch_size))
        loss_saver_noisy.append(loss_noisy/batch_size)
        loss_saver_ideal.append(loss_ideal/batch_size)
        expec_saver_noisy.append(expec_theta_noisy)
        expec_saver_ideal.append(expec_theta_ideal)
        param_saver.append(list(params))
        xi_saver.append(xi)
        fxi_saver.append(fxi)

        noisy_dens_saver.append(dm_noisy.data)
        ideal_dens_saver.append(dm_ideal.data)

    # Training is finished now test and save the test and training loss results
    if enc_circ_param is not None:
        test_metrics = test_var(hw_config=hw_config, run_config=run_config, params=params, circuits=[enc_circ_param, var_circ_param], backend=backend)
    else:
        test_metrics = None        

    train_metrics = (loss_saver_noisy,loss_saver_ideal,fidelity_saver,expec_saver_noisy,expec_saver_ideal,param_saver,xi_saver,fxi_saver)
    df_train, df_test, time_stamp = format_var_run(
        hw_config, run_config, test_metrics, train_metrics, params)

    save_var_run(hw_config, run_config, df_train, df_test, time_stamp,noisy_dens_saver,ideal_dens_saver)


def gradient_calc(j, var_circ_param, enc_circ_set, params, backend, hw_config, run_config):
    """
    j: Calc gradient for this parameter
    Gets called with partial for parallel computation of gradients
    """

    s = run_config["nbshots"]

    unit_dir = np.zeros_like(params)
    unit_dir[j] = np.pi/2
    var_circ_set_plus = var_circ_param.bind_parameters(
        params+unit_dir)
    var_circ_set_minus = var_circ_param.bind_parameters(
        params-unit_dir)
    
    if enc_circ_set is not None:
        t_circ_plus = enc_circ_set.compose(var_circ_set_plus)
        t_circ_minus = enc_circ_set.compose(var_circ_set_minus)

        t_circ_plus = custom_transpiler(t_circ_plus, backend, hw_config, run_config)
        t_circ_minus = custom_transpiler(t_circ_minus, backend, hw_config, run_config)
    else:
        t_circ_plus = var_circ_set_plus
        t_circ_minus = var_circ_set_minus

  
    order_plus = qeorder(t_circ_plus._layout,run_config["nqubits"])
    order_minus = qeorder(t_circ_minus._layout,run_config["nqubits"])

    dm_instr_plus = SaveDensityMatrix(run_config["nqubits"])
    dm_instr_minus = SaveDensityMatrix(run_config["nqubits"])

    t_circ_plus.append(dm_instr_plus, order_plus)
    t_circ_minus.append(dm_instr_minus, order_minus)

    result_plus = backend.run(t_circ_plus, shots=s).result()
    result_minus = backend.run(t_circ_minus, shots=s).result()

    dm_plus = result_plus.data()["density_matrix"]
    dm_minus = result_minus.data()["density_matrix"]

    if run_config["nqubits"] > 1:
       # Since we created the density matrix in the particular order we did, we can just use increasing numbering
        dm_red_plus = partial_trace(dm_plus,[i for i in range(run_config["measure_pos"])] + [j for j in range(run_config["measure_pos"]+1,run_config["nqubits"])])
        dm_red_minus = partial_trace(dm_minus,[i for i in range(run_config["measure_pos"])] + [j for j in range(run_config["measure_pos"]+1,run_config["nqubits"])])

    else:
        dm_red_plus = dm_plus
        dm_red_minus = dm_minus

    probs_plus = dm_red_plus.probabilities_dict()
    probs_minus = dm_red_minus.probabilities_dict() # Save to later see whether the noisy free variant would have produced better - or worse results

    if len(probs_plus.keys())==2:
        expec_plus_j = (probs_plus['0']-probs_plus['1'])
    elif '1' in probs_plus.keys():
        expec_plus_j = -1
    else:
        expec_plus_j = 1

    if len(probs_minus.keys())==2:
        expec_minus_j = (probs_minus['0']-probs_minus['1'])
    elif '1' in probs_minus.keys():
        expec_minus_j = -1
    else:
        expec_minus_j = 1
    
    return (expec_plus_j, expec_minus_j)


def run(run_config: dict, hw_config: dict):
    """
    Execute grover or iqft simulation with given configs
    """
    print("Simulating {} with {}".format(
        hw_config["hw_name"], str(run_config.items())))

    # Set noise model
    match run_config["noise_method"]:
        case "inherent":
            noise_model = hw_to_noise(hw_config)
        case "depol":
            noise_model = pauli_depol(
                hw_config=hw_config, run_config=run_config)
        case "x":
            noise_model = pauli_x(hw_config=hw_config, run_config=run_config)
        case "y":
            noise_model = pauli_y(hw_config=hw_config, run_config=run_config)
        case "z":
            noise_model = pauli_z(hw_config=hw_config, run_config=run_config)
        case "none":
            noise_model = NoiseModel()
        case other:
            raise ValueError("Please choose valid options for noise_method")

    # NOTE: The variational circuit got its own run method since its kinda different
    match run_config["algo"]:
        case "grover":
            circuit = get_grover_circuit(run_config)
        case "iqft":
            circuit = prepare_base_qft(run_config)
        case other:
            raise ValueError("Please choose on of the supported algorithms")

    backend = AerSimulator(noise_model=noise_model,method=run_config["method"], coupling_map=hw_config["coupling_map"])
    dm_ideal = DensityMatrix(circuit)
    t_circuit = custom_transpiler(circuit, backend, hw_config, run_config)


    time_estimate = estimate_circuit_time(t_circuit, tg1=hw_config["TG1_avg"], tg2=hw_config["TG2_avg"], filter_function=lambda x: not (
        'rz' == getattr(x.operation, "name") or 'z' == getattr(x.operation, "name") or 'barrier' == getattr(x.operation, "name") or 'measure' == getattr(x.operation, "name")))
    depth_estimate = t_circuit.depth(filter_function=lambda x: not (
        'rz' == getattr(x.operation, "name") or 'z' == getattr(x.operation, "name") or 'barrier' == getattr(x.operation, "name") or 'measure' == getattr(x.operation, "name")))

    # Reorder measurement order such that introduced swaps by transpilation are miltigated
    cregs = ClassicalRegister(run_config["nqubits"])
    t_circuit.add_register(cregs)
    order = qeorder(t_circuit._layout,run_config["nqubits"])

    # Attach density matrix instruction in correct order
    dmatrix_noisy_instr = SaveDensityMatrix(run_config["nqubits"])
    t_circuit.append(dmatrix_noisy_instr, order)

    # Also measure in correct order
    for cidx, qidx in enumerate(order):
        t_circuit.measure(qidx,cidx)

    result = backend.run(t_circuit, shots=run_config["nbshots"]).result()
    dm_noisy = result.data()["density_matrix"]
    df, time_stamp = format_run(hw_config=hw_config, run_config=run_config,
                                aer_result=result, time_estimate=time_estimate, depth_estimate=depth_estimate, dm_ideal=dm_ideal, dm_noisy=dm_noisy)
    save_run(hw_config=hw_config, run_config=run_config, df=df,
             time_stamp=time_stamp, dm_ideal=dm_ideal, dm_noisy=dm_noisy)

    print("Finished {} with {}".format(
        hw_config["hw_name"], str(run_config.items())))



def qeorder(lay,nqubits):
    """
    qubit-reorder:
    During transpilation with a coupling map the qubits get reorderd -->
    We need to adapt the order for measuring and attaching the density matrix instruction!
    NOTE: This is a little provisoric and could be updated in the future
    """
    # None type for full coupling graphs --> Just use default coupling
    if lay is None:
        return [i for i in range(nqubits)]
    if lay.final_layout is None:
        return [i for i in range(nqubits)]

    final_lay = lay.final_layout.get_physical_bits()
    initial_lay = lay.initial_layout.get_physical_bits()
    noAncilla = [key for key,value in initial_lay.items() if "ancilla" not in value._repr]

    ff = lay.final_layout
    p2v = ff._p2v

    key_ordered = list(p2v.keys())
    new_order = []
    for elem in noAncilla:
        new_order.append(key_ordered[elem])
        
    return new_order


def remove_qidle(t_circuit):
    """
    Little hack to remove quantum idle wires to enable calculation of the density matrix
    (NOTE: Deprecated since we now use the reordered stuff; but may still become handy someday)
    """
    qubit_names = []
    for instruction in t_circuit._data:
        for qubit in instruction.qubits:
            if qubit not in qubit_names:
                qubit_names.append(qubit)

    new_circ = QuantumCircuit(len(qubit_names))

    for instruction in t_circuit._data:
        idx = []
        for qubit in instruction.qubits:
            idx.append(qubit_names.index(qubit))
        new_circ.append(instruction.operation, qargs=idx)
    return new_circ


def format_run(hw_config, run_config, aer_result, time_estimate, depth_estimate, dm_ideal, dm_noisy):
    """
    Format results from run in pandas array
    Return: pandas array, timestamp
    TODO: Remove Special characters for easier import in the future
    """
    time_stamp = str(datetime.datetime.now())
    df = pd.DataFrame()

    # Calc fidelity (Overlap)
    state_fid = state_fidelity(dm_ideal, dm_noisy)

    winkey = np.binary_repr(
        run_config["winner_state"], width=run_config["nqubits"])
    
    if winkey in dm_noisy.probabilities_dict().keys():
        success_dm_noisy = dm_noisy.probabilities_dict()[winkey]
    else:
        success_dm_noisy = 0
    success_dm_ideal = dm_ideal.probabilities_dict()[winkey]

    print(f"-#-#-#-#-#-Succes rate dm_noisy = {success_dm_noisy}-#-#-#-#-#-")

    # Prepare data array
    df["timestamp"] = [time_stamp]
    df["algorithm"] = [run_config["algo"]]
    df["vendor"] = [hw_config["hw_name"].split("_")[0]]
    df["machine"] = [hw_config["hw_name"].split("_")[1]]
    df["method"] = [run_config["method"]]
    df["#qubits"] = [run_config["nqubits"]]
    df["#shots"] = [run_config["nbshots"]]
    df["opt_level"] = [run_config["opt_level"]]
    df["noise_method"] = [run_config["noise_method"]]
    df["noise_level"] = [run_config["noise_level"]]
    df["circ_depth"] = [depth_estimate]
    df["circ_time"] = [time_estimate]
    # df["success_rate"] = [success_rate]
    df["noisy_succes_rate"] = [success_dm_noisy]
    df["ideal_success_rate"] = [success_dm_ideal]
    df["state_fidelity"] = [state_fid]
    df["winner_state"] = [np.binary_repr(
        run_config["winner_state"], width=run_config["nqubits"])]

    return df, time_stamp


def save_run(hw_config, run_config, df, time_stamp, dm_ideal, dm_noisy):
    """
    Save data for quantum algorithms
    TODO: Remove Special characters for easier import in the future
    """
    filename = hw_config["hw_name"] + "_" + run_config["noise_method"] +"_" + time_stamp
    path = "./sim_results/"+str(run_config["algo"])
    if not os.path.isdir(path):
        os.makedirs(path)

    df.to_csv(os.path.join(path, filename+".csv"),
              encoding='utf-8', index=False)

    path_dens = "./density_results/"+str(run_config["algo"])
    if not os.path.isdir(path_dens):
        os.makedirs(path_dens)

    # We need only the noisy ones, since the ideal ones are equal for every run.
    # np.save(os.path.join(path_dens, "dm_ideal"+filename), dm_ideal)
    
    # Use timestamp in csvs to find the density matrix to your dings
    np.save(os.path.join(path_dens, "dm-noisy_"+filename), dm_noisy.data)


def format_var_run(hw_config, run_config, test_metrics, train_metrics, params):
    """
    Format results from variational circuit
    TODO: Remove Special characters for easier import in the future
    """

    # Structure Test Metrics

    loss_train_noisy, loss_train_ideal, fidelity_train,expec_train_noisy,expec_train_ideal,param_train,xi_train,fxi_train = train_metrics

    if test_metrics is not None:
        xi,fxi,f_predict_noisy,f_predict_ideal,fidelity_test = zip(*test_metrics)

    # xi, fxi, f_predict, loss_test = zip(*test_metrics)

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    time_stamp = str(datetime.datetime.now())

    train_iters = run_config["iterations"]


    # Training data frame
    df_train["train_iter"] = [i for i in range(train_iters)]
    df_train["loss_train_noisy"] = loss_train_noisy
    df_train["loss_train_ideal"] = loss_train_ideal
    df_train["fidelity_train"] = fidelity_train

    df_train["timestamp"] = [time_stamp] * train_iters
    df_train["fun"] = [run_config["fun"]] * train_iters
    df_train["vendor"] = [hw_config["hw_name"].split("_")[0]] * train_iters
    df_train["machine"] = [hw_config["hw_name"].split("_")[1]]*train_iters
    df_train["method"] = [run_config["method"]]*train_iters
    df_train["#qubits"] = [run_config["nqubits"]]*train_iters
    df_train["#shots"] = [run_config["nbshots"]]*train_iters
    df_train["#iterations"] = [train_iters]*train_iters
    df_train["eta"] = [run_config["eta"]]*train_iters
    df_train["batch_size"] = [run_config["batch_size"]]*train_iters
    df_train["opt_level"] = [run_config["opt_level"]]*train_iters
    df_train["noise_method"] = [run_config["noise_method"]]*train_iters
    df_train["noise_level"] = [run_config["noise_level"]]*train_iters
    df_train["circuit_repetitions"] = [run_config["L"]]*train_iters
    df_train["fx_pred_noisy"] = list(expec_train_noisy)
    df_train["fx_pred_ideal"] = list(expec_train_ideal)
    df_train["xi_train"] = list(xi_train)
    df_train["fxi_train"] = list(fxi_train)
    df_train["measure_pos"] = [run_config["measure_pos"]]*train_iters

    for i, param_list in enumerate(zip(*param_train)):
        df_train["theta_"+str(i)] = list(param_list)

    if test_metrics is None:
        return df_train, None, time_stamp

    # Testing data frame
    test_len = len(xi)
    df_test["x_in"] = list(xi)
    df_test["fx_pred_noisy"] = list(f_predict_noisy)
    df_test["fx_pred_ideal"] = list(f_predict_ideal)
    df_test["fx_true"] = list(fxi)
    df_test["fidelity_test"] = list(fidelity_test)
    df_test["timestamp"] = [time_stamp] * test_len
    df_test["fun"] = [run_config["fun"]] * test_len
    df_test["vendor"] = [hw_config["hw_name"].split("_")[0]] * test_len
    df_test["machine"] = [hw_config["hw_name"].split("_")[1]]*test_len
    df_test["method"] = [run_config["method"]]*test_len
    df_test["#qubits"] = [run_config["nqubits"]]*test_len
    df_test["#shots"] = [run_config["nbshots"]]*test_len
    df_test["#iterations"] = [train_iters]*test_len
    df_test["eta"] = [run_config["eta"]]*test_len
    df_test["batch_size"] = [run_config["batch_size"]]*test_len
    df_test["opt_level"] = [run_config["opt_level"]]*test_len
    df_test["noise_method"] = [run_config["noise_method"]]*test_len
    df_test["noise_level"] = [run_config["noise_level"]]*test_len
    df_test["circuit_repetitions"] = [run_config["L"]]*test_len
    df_test["measure_pos"] = [run_config["measure_pos"]]*test_len

    for i in range(len(params)):
        df_test["theta_"+str(i)] = [params[i]]*test_len

    return df_train, df_test, time_stamp


def save_var_run(hw_config, run_config, df_train, df_test, time_stamp,dms_noisy,dms_ideal):
    """
    Save data from variational circuit
    """

    filename = hw_config["hw_name"] + "_" + run_config["fun"] + "_" + run_config["noise_method"] +"=" + str(run_config["noise_level"]) +"_"+ time_stamp + ".csv"
    path_train = "./varsim_train/"+str(run_config["hw"])
    

    path_dens = "./varsim_train_density/"+str(run_config["hw"])
    filename_dens_ideal = "noisefree_" + hw_config["hw_name"] + "_" + run_config["fun"] + "_" + run_config["noise_method"] +"=" + str(run_config["noise_level"]) +"_"+ time_stamp
    filename_dens_noisy = "noisy_"+ hw_config["hw_name"] + "_" + run_config["fun"] + "_" + run_config["noise_method"] +"=" + str(run_config["noise_level"]) +"_"+ time_stamp

    if not os.path.isdir(path_dens):
        os.makedirs(path_dens)

    np.save(os.path.join(path_dens,filename_dens_noisy),dms_noisy)
    np.save(os.path.join(path_dens,filename_dens_ideal),dms_ideal)


    if not os.path.isdir(path_train):
        os.makedirs(path_train)

    df_train.to_csv(os.path.join(path_train, filename),
                    encoding='utf-8', index=False)
    
    if df_test is not None:
        path_test = "./varsim_test/"+str(run_config["hw"])
        if not os.path.isdir(path_test):
            os.makedirs(path_test)

        df_test.to_csv(os.path.join(path_test, filename),encoding='utf-8', index=False)


def test_var(hw_config, run_config, params, circuits, backend):
    """
    Tests variational circuit on some data; saves training and test results
    circuits is assumed to be a list of the not binded version of both
    """

    enc_circ_param, var_circ_param = circuits
    var_circ_set = var_circ_param.bind_parameters(params)

    test_data = []
    match run_config["fun"]:
        case "powx2":
            for x in np.arange(-1, 1, 0.1):
                test_data.append((x, x*x))
        case other:
            raise ValueError(
                "Implement other target functions in test cases..")

    test_data = sorted(test_data)
    test_metrics = []
    new_test_metrics = []
    for xi, fxi in test_data:

        enc_circ_set = enc_circ_param.bind_parameters([xi])
        t_circ = enc_circ_set.compose(var_circ_set)

        dm_ideal = DensityMatrix(t_circ)
        order = qeorder(t_circ._layout,run_config["nqubits"])

        # Attach density matrix instruction in correct order
        dmatrix_noisy_instr = SaveDensityMatrix(run_config["nqubits"])
        t_circ = custom_transpiler(t_circ, backend, hw_config, run_config)
        t_circ.append(dmatrix_noisy_instr, order)

        result = backend.run(t_circ, shots=0).result()
        dm_noisy = result.data()["density_matrix"]

        dm_red_ideal = partial_trace(dm_ideal,[i for i in range(run_config["measure_pos"])] + [j for j in range(run_config["measure_pos"]+1,run_config["nqubits"])])
        dm_red_noisy = partial_trace(dm_noisy,[i for i in range(run_config["measure_pos"])] + [j for j in range(run_config["measure_pos"]+1,run_config["nqubits"])])

        probs_noisy = dm_red_noisy.probabilities_dict()
        probs_ideal = dm_red_ideal.probabilities_dict() # Save to later see whether the noisy free variant would have produced better - or worse results

        state_fid = state_fidelity(dm_noisy,dm_ideal)

        if len(probs_noisy.keys())==2:
            expec_theta_noisy = (probs_noisy['0']-probs_noisy['1'])
        elif '1' in probs_noisy.keys():
            expec_theta_noisy = -1
        else:
            expec_theta_noisy = 1

        if len(probs_ideal.keys())==2:
            expec_theta_ideal = (probs_ideal['0']-probs_ideal['1'])
        elif '1' in probs_ideal.keys():
            expec_theta_ideal = -1
        else:
            expec_theta_ideal = 1


        print(
            f"Input = {xi} | Desired output = {fxi} | Actual output = {expec_theta_noisy} | Loss = {0.5*(expec_theta_noisy-fxi)**2}")
        
        # No loss, since we can calculate in from fix and expec theta
        new_test_metrics.append((xi,fxi,expec_theta_noisy,expec_theta_ideal,state_fid))
    return new_test_metrics



def custom_transpiler(circuit, backend, hw_config, run_config):
    """
    Transpile different backends to native gatesets
    Returns: Transpiled circuit
    """
    gates, qubits = zip(*hw_config["basis_gates"])
    gates = list(gates)
    gates.append('unitary')

    # NOTE: The extra transpile step does not seem to be necessary, since the backend.run method cares for the noise later
    match hw_config["hw_name"]:
        case 'ionq_aria':
            gates.append('rxx')  # Gates added as interim steps
            gates.append('rz')
            own_translator = IonQ_Translator()
            new_circ = own_translator.run(circuit, run_config)
            t_circ = new_circ
        case 'ibmq_kolkata'  :
            # In this case we just transpile according to the optimization level!
            t_circ = transpile(circuit, backend=backend, basis_gates=gates,
                               optimization_level=run_config["opt_level"])
        case 'rigetti_aspenm3':
            # NOTE: Also put hw_config in later for coupling graph
            own_translator = RigettiTranslator(run_config, hw_config)
            new_circ = own_translator.run(circuit)
            t_circ = new_circ
        case 'simple_simple':
            t_circ = transpile(circuit, backend=backend, basis_gates=gates,optimization_level=run_config["opt_level"])
                
        case 'ibmq_connected':
            # In this case we just transpile according to the optimization level!
            t_circ = transpile(circuit, backend=backend, basis_gates=gates,
                               optimization_level=run_config["opt_level"])
        
        case 'rigetti_connected':
            # NOTE: Also put hw_config in later for coupling graph
            own_translator = RigettiTranslator(run_config, hw_config)
            new_circ = own_translator.run(circuit)
            t_circ = new_circ

        case 'config_template':
            # In this case we just transpile according to the optimization level!
            t_circ = transpile(circuit, backend=backend, basis_gates=gates,
                               optimization_level=run_config["opt_level"])

        case other:
            raise ValueError("Please choose valid custom transpiler!")

    return t_circ

def pauli_x(hw_config, run_config):
    """
    Only Bitflip and Tensored BitFlips
    """
    if run_config["noise_level"] < 0 or run_config["noise_level"] > 1:
        raise ValueError("Choose valid noise level [0,1) for pauli noise")

    noise_model = NoiseModel()

    err_1 = pauli_error([('X', run_config["noise_level"]),
                        ('I', 1 - run_config["noise_level"])])

    # Choose two qubit gate error as simple expansion of Pauli BitFlip (Tensor product reverse; actually order does barely matter):
    err_2 = err_1.expand(err_1)

    hw_config["basis_gates"].append(("unitary", 1))
    hw_config["basis_gates"].append(("unitary", 2))

    for gate_name, gate_size in hw_config["basis_gates"]:
        if gate_name == "rz" or gate_name == "z":
            continue
        if gate_size == 2:
            coupling = hw_config["coupling_map"]
            for neighbors in coupling:
                noise_model.add_quantum_error(
                    err_2, instructions=gate_name, qubits=neighbors)
        else:
            for qubit in range(hw_config["qubit_count"]):
                noise_model.add_quantum_error(
                    err_1, instructions=gate_name, qubits=[qubit])

    noise_model.add_basis_gates(['unitary'])

    hw_config["basis_gates"].pop()
    hw_config["basis_gates"].pop()

    return noise_model

def pauli_y(hw_config, run_config):
    """
    Only Bitflip and Tensored BitFlips
    """
    if run_config["noise_level"] < 0 or run_config["noise_level"] > 1:
        raise ValueError("Choose valid noise level [0,1) for pauli noise")
    noise_model = NoiseModel()

    err_1 = pauli_error([('Y', run_config["noise_level"]),
                        ('I', 1 - run_config["noise_level"])])

    # Choose two qubit gate error as simple expansion of Pauli BitFlip (Tensor product reverse; actually order does barely matter):
    err_2 = err_1.expand(err_1)

    hw_config["basis_gates"].append(("unitary", 1))
    hw_config["basis_gates"].append(("unitary", 2))

    for gate_name, gate_size in hw_config["basis_gates"]:
        if gate_name == "rz" or gate_name == "z":
            continue
        if gate_size == 2:
            coupling = hw_config["coupling_map"]
            for neighbors in coupling:
                noise_model.add_quantum_error(
                    err_2, instructions=gate_name, qubits=neighbors)
        else:
            for qubit in range(hw_config["qubit_count"]):
                noise_model.add_quantum_error(
                    err_1, instructions=gate_name, qubits=[qubit])

    noise_model.add_basis_gates(['unitary'])

    hw_config["basis_gates"].pop()
    hw_config["basis_gates"].pop()

    return noise_model


def pauli_z(hw_config, run_config):
    """
    Only Bitflip and Tensored BitFlips
    """
    if run_config["noise_level"] < 0 or run_config["noise_level"] > 1:
        raise ValueError("Choose valid noise level [0,1) for pauli noise")

    noise_model = NoiseModel()

    err_1 = pauli_error([('Z', run_config["noise_level"]),
                        ('I', 1 - run_config["noise_level"])])

    # Choose two qubit gate error as simple expansion of Pauli BitFlip (Tensor product reverse; actually order does barely matter):
    err_2 = err_1.expand(err_1)

    hw_config["basis_gates"].append(("unitary", 1))
    hw_config["basis_gates"].append(("unitary", 2))

    for gate_name, gate_size in hw_config["basis_gates"]:        
        if gate_name == "rz" or gate_name == "z":
            continue
        if gate_size == 2:
            coupling = hw_config["coupling_map"]
            for neighbors in coupling:
                noise_model.add_quantum_error(
                    err_2, instructions=gate_name, qubits=neighbors)
        else:
            for qubit in range(hw_config["qubit_count"]):
                noise_model.add_quantum_error(
                    err_1, instructions=gate_name, qubits=[qubit])

    noise_model.add_basis_gates(['unitary'])

    hw_config["basis_gates"].pop()
    hw_config["basis_gates"].pop()

    return noise_model

def pauli_depol(hw_config, run_config):
    """
    Only depol
    """
    if run_config["noise_level"] < 0 or run_config["noise_level"] > 1:
        raise ValueError("Choose valid noise level [0,1) for pauli noise")

    noise_model = NoiseModel()
    err_1 = depolarizing_error(run_config["noise_level"], 1)
    err_2 = depolarizing_error(run_config["noise_level"], 2)

    hw_config["basis_gates"].append(("unitary", 1))
    hw_config["basis_gates"].append(("unitary", 2))

    for gate_name, gate_size in hw_config["basis_gates"]:
        if gate_name == "rz" or gate_name == "z":
            continue
        if gate_size == 2:
            coupling = hw_config["coupling_map"]
            for neighbors in coupling:
                noise_model.add_quantum_error(
                    err_2, instructions=gate_name, qubits=neighbors)
        else:
            for qubit in range(hw_config["qubit_count"]):
                noise_model.add_quantum_error(
                    err_1, instructions=gate_name, qubits=[qubit])

    noise_model.add_basis_gates(['unitary'])

    hw_config["basis_gates"].pop()
    hw_config["basis_gates"].pop()

    return noise_model

# Taken from Qiskit
def truncate_t2_value(t1, t2):
    """Return t2 value truncated to 2 * t1 (for t2 > 2 * t1)"""
    new_t2 = t2
    if t2 > 2 * t1:
        new_t2 = 2 * t1
    return new_t2

def hw_to_noise(hw_config: dict):
    """
    Create noise model that matches fidelity of given hardware
    Returns: NoiseModel
    Some parts of this code stem from / are inspired by
    qiskit_aer/noise/device/models.py
    """
    # Init empty noise model; Subsequently add quantum errors
    noise_model = NoiseModel()

    t1, t2 = hw_config["T1_avg"], hw_config["T2_avg"]
    t2 = truncate_t2_value(t1, t2)
    tg1, tg2 = hw_config["TG1_avg"], hw_config["TG2_avg"]
    e1, e2 = hw_config["e1_avg"], hw_config["e2_avg"]

    relax_errors = []
    combined_errors = []

    # NOTE: In case of custom gates; the hardware config has to include the gate type 'unitary' for the one and two qubit case
    hw_config["basis_gates"].append(("unitary", 1))
    hw_config["basis_gates"].append(("unitary", 2))

    # Create relax error for every gate
    for gate_name, gate_size in hw_config["basis_gates"]:
        if gate_size == 1:
            relax_errors.append(
                (gate_name, gate_size, thermal_relaxation_error(t1, t2, tg1)))
        elif gate_size == 2:
            temp_error = thermal_relaxation_error(t1, t2, tg2)
            relax_errors.append(
                (gate_name, gate_size, temp_error.expand(temp_error)))
        else:
            print("Choose 1 / 2 qubit gates; Return empty model")
            return noise_model

    # Create Depol error for every gate, depending on literature fidelity
    for gate_name, gate_size, relax_error in relax_errors:
        if gate_name == "rz" or gate_name == "z":
            continue
        relax_fid = average_gate_fidelity(relax_error)
        relax_infid = 1 - relax_fid
        dim = 2**gate_size
        if gate_size == 1:
            error_param = hw_config["e1_avg"]
        else:
            error_param = hw_config["e2_avg"]

        if relax_infid > error_param:
            raise ValueError(
                "The relaxation fidelity exceeds the targeted already - Abort Simulation.")

        # Here the composition of depol and thermal relaxation error is constructed.
        # The code idea steams from qiskit, the explanation is formulated out in our paper

        depol_param = dim * (error_param - relax_infid) / (dim * relax_fid - 1)
        depol_error = depolarizing_error(depol_param, num_qubits=gate_size)
        comb_error = depol_error.compose(relax_error)
        combined_errors.append((gate_name, gate_size, comb_error))

    for gate_name, gate_size, error in combined_errors:
        if gate_size == 2:
            coupling = hw_config["coupling_map"]
            for neighbors in coupling:
                noise_model.add_quantum_error(
                    error, instructions=gate_name, qubits=neighbors)
        else:
            for qubit in range(hw_config["qubit_count"]):
                noise_model.add_quantum_error(
                    error, instructions=gate_name, qubits=[qubit])

    noise_model.add_basis_gates(
        ['unitary'])

    # Delete unitary type again from gate def
    hw_config["basis_gates"].pop()
    hw_config["basis_gates"].pop()

    return noise_model


if __name__ == "__main__":
    config_list = parse_args()

    main(config_list)
