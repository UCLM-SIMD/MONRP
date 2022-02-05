from abc import ABC, abstractmethod
import copy
import random
from typing import Any, Dict, List, Tuple
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt

from models.Solution import Solution
from algorithms.abstract_algorithm.evaluation_exception import EvaluationLimit

from datasets.Dataset import Dataset


class AbstractAlgorithm(ABC):
    """Abstract class for algorithm implementations
    """

    def __init__(self, dataset_name: str = "1", random_seed: int = None, debug_mode: bool = False, tackle_dependencies: bool = False):
        """Default init method that sets common arguments such as dataset, seed and modes.

        Args:
            dataset_name (str, optional): [description]. Defaults to "1".
            random_seed (int, optional): [description]. Defaults to None.
            debug_mode (bool, optional): [description]. Defaults to False.
            tackle_dependencies (bool, optional): [description]. Defaults to False.
        """
        self.dataset: Dataset = Dataset(dataset_name)
        self.dataset_name: str = dataset_name

        self.debug_mode: bool = debug_mode
        self.tackle_dependencies: bool = tackle_dependencies
        self.random_seed: int = random_seed
        self.set_seed(random_seed)

        self.nds_debug = []
        self.population_debug = []

    def set_seed(self, seed: int):
        self.random_seed: int = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def reset(self) -> None:
        """Method that clears specific algorithm implementation variables and data
        """
        self.nds_debug = []
        self.population_debug = []

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Main method of algorithm that runs the search

        Returns:
            Dict[str, Any]: results of execution are returned as a dict
        """
        pass

    def generate_chart(self, plot) -> Tuple[List[float], List[float]]:
        """Aux method that runs the algorithm and plots the returned data
        """
        result = self.run()
        func = [i for i in result["population"]]
        function1 = [i.total_satisfaction for i in func]
        function2 = [i.total_cost for i in func]
        plot.scatter(function2, function1, label=self.get_name())
        return function1, function2

    @abstractmethod
    def get_file(self)->str:
        pass

    def dependencies_to_string(self)->str:
        return "deps" if self.tackle_dependencies else "no_deps"

    def get_name(self) -> str:
        return "Algorithm"

    @abstractmethod
    def stop_criterion(self) -> bool:
        pass

    def evaluate(self, population: List[Solution], best_individual: Solution) -> Solution:
        try:
            best_score = 0
            new_best_individual = None
            for ind in population:
                ind.evaluate()
                if ind.mono_objective_score > best_score:
                    new_best_individual = copy.deepcopy(ind)
                    best_score = ind.mono_objective_score
                self.add_evaluation(population)
            if best_individual is not None:
                if new_best_individual.mono_objective_score > best_individual.mono_objective_score:
                    best_individual = copy.deepcopy(new_best_individual)
            else:
                best_individual = copy.deepcopy(new_best_individual)
        except EvaluationLimit:
            pass
        return best_individual

    def repair_population_dependencies(self, solutions: List[Solution]) -> List[Solution]:
        for sol in solutions:
            sol.correct_dependencies()
        return solutions

    def createGIF(self, input_folder: str = "temp", output_filename: str = "example_gif", dpi: int = 100, fps: int = 1,
                  onlyNDS: bool = False, max_x: float = 1.0, max_y: float = 1.0) -> None:
        plt.rcParams['figure.figsize'] = [16, 10]
        plt.rcParams['figure.dpi'] = 200

        alg_results = []
        filenames = []

        self.debug_mode = True

        result = self.run()

        alg_results.append(result["nds_debug"])
        alg_results.append(result["population_debug"])

        # loop pareto steps and generate a frame with all points for all algorithms
        for pareto_index in range(len(alg_results[0])):
            plt.cla()
            plt.clf()

            # intermediate pareto results for a frame
            func = [j for j in alg_results[0][pareto_index]]
            functiony = [i.total_satisfaction for i in func]
            functionx = [i.total_cost for i in func]
            plt.scatter(functionx, functiony,
                        label="NDS")

            if not onlyNDS:
                # intermediate population results for a frame
                func = [j for j in alg_results[1][pareto_index]]
                functiony = [i.total_satisfaction for i in func]
                functionx = [i.total_cost for i in func]
                plt.scatter(functionx, functiony,
                            label="Population")

            # config frame
            plt.xlim([0, max_x])
            plt.ylim([0, max_y])
            plt.xlabel('Effort', fontsize=12)
            plt.ylabel('Satisfaction', fontsize=12)
            plt.legend(loc="lower right")
            plt.title(
                f'{self.get_name()} Dataset {str(self.dataset_name)} It={str(pareto_index)}')
            plt.grid(True)
            plt.draw()
            # store frame
            filename = f'{input_folder}/temp{str(pareto_index+1)}.png'
            filenames.append(filename)
            plt.savefig(filename, dpi=dpi)

        # Build GIF
        print('Creating gif\n')
        with imageio.get_writer(f'{output_filename}.gif', mode='I', fps=fps) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            for i in range(fps*5):
                writer.append_data(image)

        print('Gif saved\n')
        print('Removing Images\n')
        # Remove files
        for filename in set(filenames):
            os.remove(filename)
        print('DONE')

    def debug_data(self, nds_debug=None, population_debug=None):
        if nds_debug is not None:
            self.nds_debug.append(nds_debug.copy())
        else:
            self.nds_debug.append(self.nds.copy())

        if population_debug is not None:
            self.population_debug.append(population_debug.copy())
        else:
            self.population_debug.append(self.population.copy())
