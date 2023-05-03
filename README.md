# MONRP

![Python 3.8](https://img.shields.io/badge/Python-3.8.8-blue)

# First steps

First, install dependencies: `pip install -r requirements.txt`

# Datasets

Datasets are .json files, available at https://figshare.com/s/c9c81f3c6e01a0423e26

---


# Algorithms

All algorithms inherit the base class:

```python
Algorithm(dataset_name:str="1", random_seed:int=None, debug_mode:bool=False, tackle_dependencies:bool=False)
```

Common parameters for all algorithms are:

- `dataset_name`: for the specific dataset to be loaded
- `random_seed`: for deterministic executions
- `debug_mode`: to allow the algorithm to save intermediate results for further debugging of the execution
- `tackle_dependencies`: to solve the problem taking into account interactions between requirements.

Common methods for all algorithms are:

- `set_seed(seed)`: for setting random seed
- `reset()`: for clearing algorithm values after execution
- `run()`: for running the algorithm
- `get_name()`: to obtain its descriptive name
- `stop_criterion()`: defines when the execution should stop

Current algorithm families are the following:

- [Genetic (base)](#genetic-abstract)
  - [GeneticNDS](#geneticnds)
 
- [EDA](#eda)

  - [UMDA](#umda-univariate)
  - [PBIL](#pbil-univariate)
  - [MIMIC](#feda-univariate)
  

  ***

## Genetic (abstract)

Implements common operators of genetic algorithms, such as initialization, basic selection, crossover, mutation and replacement operators. Specific implementations may override or use new operators.

Specific parameters for genetic algorithms are the following:

```python
population_length: int,
max_generations: int,
max_evaluations: int,
selection: str,
selection_candidates: int,
crossover: str,
crossover_prob: float,
mutation: str,
mutation_prob: float,
replacement: str,
```

### GeneticNDS

Single-Objective genetic algorithm that updates a set of nondominated solutions after each generation.



## EDA

Estimation of Distribution Algorithms.

Specific parameters for EDA are the following:

```python
population_length: int,
max_generations: int,
max_evaluations: int,
```

Common methods for EDAs are:

- `generate_initial_population()`
- `select_individuals()`:
- `learn_probability_model()`:
- `sample_new_population()`:

### UMDA (univariate)

Univariate Marginal Distribution Algorithm.

Specific implementations for UDMA are:

- `learn_probability_model()`:
- `sample_new_population()`:


### PBIL (univariate)

Population Based Incremental Learning

Specific implementations for PBIL are:

- `select_individuals()`:
- `learn_probability_model()`:
- `sample_new_population()`:

Specific methods for PBIL are:

- `initialize_probability_vector()`:


### Visual indicators (scatter plots) of experiments with 12 datasets and 4 algorithms (SOGA, UMDA, PBIL, MIMIC, NSGA-II and HMORG)

Dataset p1
![scatter_p1](https://user-images.githubusercontent.com/25950319/197763505-edc9f355-5c5c-4606-a0cf-3ca874a6efc8.svg)
Dataset p2

Dataset s1

Dataset s2

Dataset s3

Dataset s4






