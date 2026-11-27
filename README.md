# Photonic STDP simulation

_NOTE: The simulation scripts, written in Python, should be cross-platform, but are only tested on Windows 10._

_NOTE: All simulations scripts where developed and tested using Python 3.15. They may or may not work as intended using other Python versions._

Simulation of time difference detection and weight update calculation for neural networks implemented in photonic hardware.

## Installation

It is not necessary but strongly recommended to install the project in a virtual environment.

To create a virtual environment, run:
```
.\create_venv.ps1
```

To activate the virtual environment, run:
```
.\activate_venv.ps1
```

To install the project and all of its dependencies, run:
```
.\install_project.ps1
```

To deactivate the virtual environment, run:
```
.\deactivate_venv.ps1
```

## Dependencies

All python dependencies are listed in the `requirements.txt` file.


## Usage

Before running a simulation, activate the python virtual environment:

```
.\activate_sim_venv
```

When the python virtual environment is activated, run the python script corresponding to the simulations from the project root directory, passing all the options as command line arguments. For example:

```
python .\scripts\SOA_simulation_Li_2008.py [options]
```

If the python virtual environment is no longer needed, either enter `deactivate` in the terminal or run the following script from the repository's root directory:

```
.\deactivate_sim_venv
```

### SOA_simulation_Li_2008_two_pulses_sweep.py

This scripts simulates multiple sweeps of of two pulses propagating through the SOA, where each iteration of each sweep increases the time bewteen the two pulses.

The simulations in the script are organized as follows:
 - There are 1 or more cases. Each case can have its own SOA and simulation configurations.
 - Each case performs a sweep simulations of 2 propagationg pulses with the same SOA and simulation configurations but with different temporal separation between pulses. The sweeps may consist of 1 or more iterations.

To execute the simulation, run:

```
python .\scripts\SOA_simulation_Li_2008_two_pulses_sweep.py [OPTIONS]
```

Many parameters of the SOA and the simulation itself can be controlled through a config file or through CLI arguments. To display a help message with all arguments available:

```
python .\scripts\SOA_simulation_Li_2008_two_pulses_sweep.py -h
```

An example config file is located in `.\sim_configs\SOA_simulation_Li_2008_two_pulses_sweep_example.toml`. To execute the simulation with this example configuration file, run:

```
python SOA_simulation_Li_2008_two_pulses_sweep.py --config-file sim_configs\SOA_simulation_Li_2008_two_pulses_sweep_example.toml
```