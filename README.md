# Estimation of Distribution Algorithms Applied to the Next Release Problem

## 17th International Conference on Soft Computing Models in Industrial and Environmental Applications (SOCO 2022)

<p align="start">
  <img src="https://img.shields.io/static/v1?label=python&message=v3.8.8&color=blue">
  <a href="https://github.com/UCLM-SIMD/MONRP/tree/soco22/datasets"><img src="https://img.shields.io/static/v1?label=datasets&message=repo&color=orange"></a>
  <a href="https://doi.org/10.1007/978-3-031-18050-7_10"><img src="https://img.shields.io/static/v1?label=conference&message=SOCO22&color=purple"></a>
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

  - [Single-Objective GA](algorithms/genetic/genetic_nds/geneticnds_algorithm.py)
  - [NSGA-II (Nondominated Sorting Genetic Algorithm II)](algorithms/genetic/nsgaii/nsgaii_algorithm.py)

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

---

## Results

Analysis with 3 datasets and 5 algorithms (Random, SOGA, NSGA-II, UMDA, and PBIL): [analysis](experimentation_analysis.ipynb) and [pairwise comparison](pairwisecomparison.ipynb).

Outputs of experiments: [outputs](output).

## Experiments

First, arrays of hyperparameter configurations have to be defined in `galgo/<algorithm...>/configurations_<algorithm>.py`

Then, use script `galgo_monrp.sh` to launch the experiments. It can generate metrics of the execution or a pareto for each configuration. Execution examples are the following:

`> sh galgo_monrp.sh metrics grasp`

`> sh galgo_monrp.sh pareto umda`

Pareto execution will generate one file for each configuration, containing one row of X-Y values for each solution found in the pareto.

Metrics execution will calculate metrics of 10 consecutive executions for each configuration, storing each configuration in a file. For grouping of these metrics, `galgo/galgo_file_merger.py` can be used. Running `galgo_file_merger.py -a <algorithm>` will generate a new output file with the column names of the metrics and will include all outputs of the given algorithm.
