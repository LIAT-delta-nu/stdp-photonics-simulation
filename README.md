# Photonic STDP simulation

_NOTE: The simulation scripts, written in Python, should be cross-platform, but are only tested on Windows 10._

_NOTE: All simulations scripts where developed and tested using Python 3.15. They may or may not work as intended using other Python versions._

## Installation


To create a python virtual environment with all the necessary dependencies:
```
.\create_sim_venv.ps1
```

## Usage

Before running a simulation, activate the python virtual environment:

```
.\activate_sim_venv
```

When the python virtual environment is activated, run the python script corresponding to the simulations. For example:

```
python .\analitic\SOA_simulation_Li_2008.py
```

If the python virtual environment is no longer needed, either enter `deactivate` in the terminal or run the following script from the repository's root directory:

```
.\deactivate_sim_venv
```

## Dependencies

All python dependencies are listed in the `requirements.txt` file.