# automated_gadget_discovery

This repository accompanies the paper "Automated gadget discovery in Science".

## Runfiles

### Quantum Optics Environment

To reproduce the results for the quantum optics environment, run the command line below which automatically loads the configuration file with the corresponding hyperparameters.

```python run_qo_ddqn.py --run {0->9}```

Then, run the following run file with the default hyperparameter settings. 

```pyhton run_qo_clustering.py```

After running the files above to generate the data, Table 4 and Table 5 can be reproduce by running the following data analysis file. This file saves the tables as csv-files in the results directory in the corresponding subfolder “analysis-and-plots”.

```pyhton run_qo_analysis.py```

To try different hyperparameter settings, a copy of the config file exp_0 can be adapted and saved in the same folder under the name exp 1 and then run with the following command:  

```python run_qo_ddqn.py --num_config 1```

### Quantum Information Environment

## Directories

```agents```

This directory contains the code for the agents: the MCTS and DDQN agent. The code of the DDQN agent is derived from the repository “”. 

```clustering```

This directory contains the code for clustering the gadgets. 

```configurations```

The files containing the parameter configuration to reproduce the results are stored in this directory. 

```data_mining```

This directory contains all the code for the gadget mining. 

```results```

The results produced by any of the runfiles are saved in this directory.
