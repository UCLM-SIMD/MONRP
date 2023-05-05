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
![linePareto0_p1](https://user-images.githubusercontent.com/25950319/236412686-f5b64e24-c9c3-4568-9ff4-ac3fb3b19be4.svg)
Dataset p2
![linePareto0_p2](https://user-images.githubusercontent.com/25950319/236412689-5371e597-ffe1-4919-a92f-e88a302cb0a0.svg)
Dataset s1
![linePareto0_s1](https://user-images.githubusercontent.com/25950319/236412692-9cc7132a-30b1-4db3-a296-a2c1328365fc.svg)
Dataset s2
![linePareto0_s2](https://user-images.githubusercontent.com/25950319/236412695-c7a1daef-6306-4220-8751-f8363234a871.svg)
Dataset s3
![linePareto0_s3](https://user-images.githubusercontent.com/25950319/236412697-36a598cb-2c04-43a2-8f24-701e1b2b09eb.svg)
Dataset s4
![linePareto0_s4](https://user-images.githubusercontent.com/25950319/236412698-5a37843c-c3a5-480a-9a59-2eed0909df5d.svg)















