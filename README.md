# MONRP

![Python 3.8](https://img.shields.io/badge/Python-3.8.8-blue)

# First steps

First, install dependencies: `pip install -r requirements.txt`

# Datasets

Datasets are .json files, described at: https://doi.org/10.5281/zenodo.7247877

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
  - [FEDA](#feda-univariate)
  

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

### FEDA (univariate)

Univariate Marginal Distribution Algorithm.

Specific implementations for FEDA are:

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


### Visual indicators (scatter plots) of experiments with 12 datasets and 4 algorithms (GA, UMDA, PBIL and FEDA)

Dataset a1
![scatter_a1](https://user-images.githubusercontent.com/25950319/197763461-cfcea645-efbc-42c9-95a0-b5f49501d817.svg)
Dataset a2
![scatter_a2](https://user-images.githubusercontent.com/25950319/197763479-b07507d6-99c1-4c6a-ac71-e497edb4c125.svg)
Dataset a3
![scatter_a3](https://user-images.githubusercontent.com/25950319/197763483-f5923cf8-cc29-4bd8-9e70-4bc5e15bb263.svg)
Dataset a4
![scatter_a4](https://user-images.githubusercontent.com/25950319/197763485-9fb89c94-b165-4044-a015-a4e1a9146ef2.svg)

Dataset c1
![scatter_c1](https://user-images.githubusercontent.com/25950319/197763486-1aa8c9ab-0473-4b85-a49e-bb28f24e015d.svg)
Dataset c2
![scatter_c2](https://user-images.githubusercontent.com/25950319/197763490-f043a239-02c2-460f-8556-3ff014e9edc4.svg)
Dataset c3
![scatter_c3](https://user-images.githubusercontent.com/25950319/197763492-8e5cbcc7-a408-43ff-86a3-23f4d5359adf.svg)
Dataset c4
![scatter_c4](https://user-images.githubusercontent.com/25950319/197763494-19391b67-1664-4f15-90c8-f43534d368db.svg)

Dataset d1
![scatter_d1](https://user-images.githubusercontent.com/25950319/197763496-ef6468e7-9998-46e2-96f0-76a74589d1c3.svg)
Dataset d2
![scatter_d2](https://user-images.githubusercontent.com/25950319/197763497-323d6144-17bc-42c4-8268-8721e112ec0d.svg)
Dataset d3
![scatter_d3](https://user-images.githubusercontent.com/25950319/197763499-fdec5b14-a94d-4ad3-809d-b22615788074.svg)
Dataset d4
![scatter_d4](https://user-images.githubusercontent.com/25950319/197763503-522dda1c-c647-44d3-a766-6ca0f9d842e5.svg)

Dataset p1
![scatter_p1](https://user-images.githubusercontent.com/25950319/197763505-edc9f355-5c5c-4606-a0cf-3ca874a6efc8.svg)
Dataset p2
![scatter_p2](https://user-images.githubusercontent.com/25950319/197763508-9529e472-c319-4029-840a-77bd45104c30.svg)


