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
![linePareto0_p1](https://user-images.githubusercontent.com/25950319/235906132-6e81f780-9f4e-437d-87ba-d0ee3bce873d.svg)
Dataset p2
![linePareto0_p2](https://user-images.githubusercontent.com/25950319/235906167-984deb9b-6d35-4508-920d-196c4f397d65.svg)
Dataset s1
![linePareto0_s1](https://user-images.githubusercontent.com/25950319/235906170-73b7ecf1-2339-470c-b196-e513225bc9a8.svg)
Dataset s2
![linePareto0_s2](https://user-images.githubusercontent.com/25950319/235906173-68eb1918-55c5-4c77-bf49-1911bb43e890.svg)
Dataset s3
![linePareto0_s3](https://user-images.githubusercontent.com/25950319/235906177-198d8d19-9b24-480d-a03d-1a6427b929da.svg)
Dataset s4
![linePareto0_s4](https://user-images.githubusercontent.com/25950319/235906178-7d7e5e54-7c32-4f5e-bc35-b19e87f20e99.svg)









