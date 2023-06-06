# FEDA-NRP: a Fixed-Structure Multivariate Estimation of Distribution Algorithm to Solve the Multi-Objective Next Release Problem with Requirements Interactions

## Journal of Engineering Applications of Artificial Intelligence

<p align="start">
  <img src="https://img.shields.io/static/v1?label=python&message=v3.8.8&color=blue">
  <a href="https://doi.org/10.5281/zenodo.7247877"><img src="https://img.shields.io/static/v1?label=datasets&message=zenodo&color=orange"></a>
  <a href="#"><img src="https://img.shields.io/static/v1?label=journal&message=EAAI&color=purple"></a>
</p>

## How to setup

Install dependencies: `pip install -r requirements.txt`

## Algorithms

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

---

### Algorithm families

- [Genetic (base)](algorithms/genetic/abstract_genetic/abstract_genetic_algorithm.py): Implements common operators of genetic algorithms, such as initialization, basic selection, crossover, mutation and replacement operators. Specific implementations may override or use new operators.

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

  Algorithms used in the study are:

  - [AGE-MOEA-II](algorithms/genetic/agemoea2/agemoea2_algorithm.py): uses PyMOO's [implementation](https://pymoo.org/algorithms/moo/age.html).
  - [C-TAEA](algorithms/genetic/ctaea/ctaea_algorithm.py): uses PyMOO's [implementation](https://pymoo.org/algorithms/moo/ctaea.html).

---

- [Estimation of Distribution Algorithms (EDA)](algorithms/EDA/eda_algorithm.py): Specific parameters for EDA are the following:

  ```python
  population_length: int,
  max_generations: int,
  max_evaluations: int,
  ```

  Common methods for EDAs are:

  `generate_initial_population()`

  `select_individuals()`

  `learn_probability_model()`

  `sample_new_population()`

  Algorithms used in the study are:

  - [UMDA (Univariate Marginal Distribution Algorithm)](algorithms/EDA/UMDA/umda_algorithm.py)
  - [PBIL (Population Based Incremental Learning)](algorithms/EDA/PBIL/pbil_algorithm.py)
  - [MIMIC](algorithms/EDA/bivariate/MIMIC/mimic_algorithm.py)
  - [FEDA (Fixed-Structure Multivariate Estimation of Distribution Algorithm)](algorithm/EDA/FEDA/feda_algorithm.py)

---

## Visual indicators (scatter plots)

Experiments with 14 datasets and 6 algorithms (UMDA, PBIL, MIMIC, AGE-MOEA, C-TAEA, and FEDA):

### Datasets PX (public datasets)

Dataset p1
![linePareto0_p1](https://github.com/UCLM-SIMD/MONRP/assets/25950319/103be913-45a5-4a3f-a1ea-6f6c7c888714)
Dataset p2
![linePareto0_p2](https://github.com/UCLM-SIMD/MONRP/assets/25950319/7360ddb6-86a6-443e-bcff-4e22233c4a0a)

### Datasets AX (agile datasets)

Dataset a1
![linePareto0_a1](https://github.com/UCLM-SIMD/MONRP/assets/25950319/e0ef85be-9c4b-4f99-9f6b-a27afca55ab6)
Dataset a2
![linePareto0_a2](https://github.com/UCLM-SIMD/MONRP/assets/25950319/229128e1-cb59-4df4-8723-94343a14b302)
Dataset a3
![linePareto0_a3](https://github.com/UCLM-SIMD/MONRP/assets/25950319/83394b98-e4d9-4280-bb7d-811397226e96)
Dataset a4
![linePareto0_a4](https://github.com/UCLM-SIMD/MONRP/assets/25950319/fa412b1f-b043-4592-82d8-ac2d35220dcc)

### Datasets CX (classic/plan-driven datasets)

Dataset c1
![linePareto0_c1](https://github.com/UCLM-SIMD/MONRP/assets/25950319/cf1ef20f-5e0a-4753-bffd-ce0b5d8401b8)
Dataset c2
![linePareto0_c2](https://github.com/UCLM-SIMD/MONRP/assets/25950319/0cc250ff-a3fd-4c57-b3ee-bfddbbacd045)
Dataset c3
![linePareto0_c3](https://github.com/UCLM-SIMD/MONRP/assets/25950319/47766574-d990-464d-8abb-f4c5a8215645)
Dataset c4
![linePareto0_c4](https://github.com/UCLM-SIMD/MONRP/assets/25950319/86b77cf9-7881-40c0-8c0b-d85db1e08840)

### Datasets DX (higher complexity classic/plan-driven datasets)

Dataset d1
![linePareto0_d1](https://github.com/UCLM-SIMD/MONRP/assets/25950319/34860c23-1e4b-4db2-a918-bce1a67d339b)
Dataset d2
![linePareto0_d2](https://github.com/UCLM-SIMD/MONRP/assets/25950319/608b0480-7ba3-4a3e-ae5b-67967ac16cc8)
Dataset d3
![linePareto0_d3](https://github.com/UCLM-SIMD/MONRP/assets/25950319/e670f16b-e6b5-44e0-a941-a5ad90f9ad00)
Dataset d4
![linePareto0_d4](https://github.com/UCLM-SIMD/MONRP/assets/25950319/c600b42f-64dc-4919-be7a-ba0142cc5ed9)
