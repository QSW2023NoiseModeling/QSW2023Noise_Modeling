# Example for executing grover
# python3 run_simulation.py --hw ionq_aria --noise_method inherent --noise_level -1 grover --par --nqubits 3 4 --parallel_reps 1
# Template Example for IQFT
# python3 run_simulation.py --hw ionq_aria --noise_method x --noise_level 0.01 iqft --par --nqubits 4 5 --parallel_reps 1
# # Template for the varational circuit:
python3 run_simulation.py --hw ibmq_kolkata --noise_method depol --noise_level 0.02 var --eta 1 --nbshots 0 --par --nqubits 4 --fun powx2 --iters 10