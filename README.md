# First steps
 
First, install dependencies: ```pip install -r requirements.txt```



# Datasets
Each dataset is placed in a single function inside a file titled ```dataset_X.py```. The function returns the following data: ```pbis_cost, num_pbis, stakeholders_importances, stakeholders_pbis_priorities, dependencies```.

Each dataset is loaded inside ```Dataset.py```, which loads and calculates scaled values and dependencies for the given data.

---

# Evaluation
Shared methods for common operations:
- ```metrics.py``` contains a set of metric calculations for evaluating an algorithm execution.
- ```update_nds.py``` provides a general method for obtaining the set of nondominated solutions of a list, given a new list of candidates.

---

# Algorithms
All algorithms inherit the base class:
```python 
Algorithm(dataset_name:str="1", random_seed:int=None, debug_mode:bool=False, tackle_dependencies:bool=False)
```
Common parameters for all algorithms are: 
- ```dataset_name```: for the specific dataset to be loaded
- ```random_seed```: for deterministic executions
- ```debug_mode```: to allow the algorithm to save intermediate results for further debugging of the execution
- ```tackle_dependencies```: to solve the problem taking into account interactions between requirements.

Common methods for all algorithms are:
- ```set_seed(seed)```: for setting random seed
- ```reset()```: for clearing algorithm values after execution
- ```run()```: for running the algorithm
- ```get_name()```: to obtain its descriptive name
- ```stop_criterion()```: defines when the execution should stop

Current algorithm families are the following:
- [Genetic (base)](#genetic-abstract)
    - [GeneticNDS](#geneticnds)
    - [NSGA-II](#nsga-ii)
- [GRASP](#grasp)
- [EDA](#eda)
    - [UMDA](#umda-univariate)
    - [PBIL](#pbil-univariate)
    - [MIMIC](#mimic-bivariate)


 ---


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

### NSGA-II
Multi-objective genetic algorithm that implements a set of operators that tend to find Pareto fronts better distributed along the search space.

Specific methods are:
- ```selection_tournament```: 
- ```replacement_elitism()```: 
- ```fast_nondominated_sort()```: 
- ```calculate_crowding_distance()```: 
- ```crowding_operator()```: 

 ---

## GRASP
Greedy Randomized Adaptive Search Procedure. Metaheuristic algorithm that iteratively constructs random solutions and by means of local search methods, tries to improve them.

Specific parameters for GRASP are the following:
```python
iterations: int, 
solutions_per_iteration: int, 
max_evaluations: int,
init_type:str,
local_search_type:str,
path_relinking_mode:str,
```

 ---

## EDA
Estimation of Distribution Algorithms.

Specific parameters for EDA are the following:
```python
population_length: int, 
max_generations: int,
max_evaluations: int,
```

Common methods for EDAs are:
- ```generate_initial_population()```
- ```select_individuals()```: 
- ```learn_probability_model()```: 
- ```sample_new_population()```:

### UMDA (univariate)
Univariate Marginal Distribution Algorithm.

Specific implementations for UDMA are:
- ```learn_probability_model()```: 
- ```sample_new_population()```: 

### PBIL (univariate)
Population Based Incremental Learning

Specific implementations for PBIL are:
- ```select_individuals()```: 
- ```learn_probability_model()```: 
- ```sample_new_population()```: 


Specific methods for PBIL are:
- ```initialize_probability_vector()```: 

### MIMIC (bivariate)

Specific parameters for MIMIC are the following:
```python
selected_individuals: int, 
selection_scheme: str,
replacement_scheme: str,
```

Specific implementations for MIMIC are:
- ```select_individuals()```: 
- ```learn_probability_model()```: 
- ```sample_new_population()```: 

---

# Testing
Test suite is placed in ```unit_tests.py``` in the root folder and can be run by executing in the cmd: 
```cmd
python -m unittest unit_tests.py
```
The steps to add new unit tests are the following:
1. Create a new test file named ```test_algorithm_name.py``` inside the algorithm folder.
2. Write a new test case class using ```unittest``` library:
    ```python
    import unittest
    from algorithms.specific_algorithm import SpecificAlgorithm as tested_algorithm_class

    class SpecificAlgorithmTestCase(unittest.TestCase):

        def setUp(self):
            """  
            Set up algorithm and random seed
            """
            seed = 0
            self.algorithm = tested_algorithm_class()
            self.algorithm.set_seed(seed)
    ```

3. Write test methods that check code is working properly:
    ```python
    def test_something(self):
        """  
        Test that `something()` method works
        """
        expected_something = 4
        actual_something = something()
        self.assertEqual(actual_something, expected_something)
    ```
4. Add the test case to ```unit_tests.py```: 
    ```python
    # import test case
    from algorithms.specific_algorithm import SpecificAlgorithmTestCase
    ...
    # add test case to the test suite
    suite.addTests(loader.loadTestsFromModule(SpecificAlgorithmTestCase))
    ...
    ```








