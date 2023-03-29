# Effects of Imperfections on Quantum Algorithms: A Software Engineering Perspective

## Prerequisities (Machine Setup)
- Install of python 3.10.6
- Ubuntu 22.04

## Install
0. Clone this repository
1. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install requirementes
```
pip install -r requirements.txt
```
## Project structure
- ./backend_data/
Contains hardware configs as numpy save
- preprocess_data.ipynb
View existing or add own low level metrics to simulate (includes explanaiton)
- run_simulation.py
Contains main methods for running the simulations
- helper_functions.py
Includes series of helper functions
- pass_defs.py
Contains transpiler passes to get custom gates running
- gate_defs.py
Contains custom gates definitions
- exec_commands.sh
Contains example CLI commands you can use to write own shell scripts

## Run simulations 
Before running: Execute the scripts in preprocess data
to adjust to your needs.
Simulations are started using the run_simulation.py script
You can use the CLI iterface or start the run simulation script
in our IDE of choice; in this case you have to set
the run_configuration manually (See the comment in def main(), in run_simulation.py)


## Additional Remarks
The CR version of this paper may contain convenience features such as a containerized runtime environments or changes
to the interface. Also the performance might be improved making the transpiler passes more efficient or
improve the current parallelization solution. 

## Acknowledgements
Next to Qiskit, we thank the creators of various useful python libraries, most importantly numpy and pandas.

## License
[Apache License 2.0](LICENSE.txt)

In the files, we used the SPDX standard for easy License identification https://spdx.dev/ ,
where we also explicitely higlight copyright holders and adjustments made, to prior work.
While reduandant, we also include the longer version, where we derived code from other sources.
We may include the redundancy in the future, when SPDX is more well adopted