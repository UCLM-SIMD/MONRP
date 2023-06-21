import json
import warnings
from abc import ABC
from typing import List

from algorithms.abstract_algorithm.abstract_algorithm import AbstractAlgorithm, plot_solutions
import evaluation.metrics as metrics
from evaluation import solution_subset_selection
from models.Solution import Solution


class AbstractExecuter(ABC):
    """Executer class used to delegate configuration, execution and formatting of algorithm outputs
    """

    def __init__(self, algorithm: AbstractAlgorithm, num_execs: int):
        """All executers store default config and metrics fields. Specific implementations might include more fields
        """
        self.executions = int(num_execs)
        self.algorithm: AbstractAlgorithm = algorithm
        self.config_fields: List[str] = ["Dataset", "Algorithm"]
        self.metrics_fields: List[str] = ["Time(s)", "HV", "Spread", "NumSolutions", "Spacing",
                                          "Requirements per sol", "AvgValue", "BestAvgValue", ]

        self.metrics_dictionary = {
            'time': [None] * self.executions,
            'nds_update_time': [None] * self.executions,
            'sss_total_time': [None] * self.executions,
            'NDS_size': [None] * self.executions,
            'HV': [None] * self.executions,
            'spread': [None] * self.executions,
            'numSolutions': [None] * self.executions,
            'spacing': [None] * self.executions,
            'mean_bits_per_sol': [None] * self.executions,
            'avgValue': [None] * self.executions,
            'bestAvgValue': [None] * self.executions,
            'gdplus': "GD+ not calculated yet. You may need to run extract_postMetrics.py",
            'unfr': "UNFR not calculated yet. You may need to run extract_postMetrics.py"

        }

    def execute(self, output_folder: str) -> None:
        """Method that executes the algorithm a number of times and saves results in json  output file
        """
        paretos_list = []  # list of pareto lists, one pareto per execution
        for it in range(0, self.executions):
            self.algorithm.reset()
            result = self.algorithm.run()

            self.metrics_dictionary['NDS_size'][it] = len(
                result['population'])  # store original NDS size created in search
            result['population'] = solution_subset_selection.search_solution_subset(sss_type=self.algorithm.sss_type,
                                                                                    subset_size=self.algorithm.subset_size,
                                                                                    solutions=result['population'])

            self.get_metrics_fields(result, it)
            pareto = self.get_pareto(result['population'])  # get a list with pareto points
            paretos_list.insert(len(paretos_list), pareto)

        #  add/update results in json output file
        self.algorithm.config_dictionary['num_executions'] = self.executions
        unique_id = ''.join(str(c) for c in self.algorithm.config_dictionary.values())
        results_dictionary = {'parameters': self.algorithm.config_dictionary,
                              'metrics': self.metrics_dictionary,
                              'paretos': paretos_list,
                              'Reference_Pareto': 'Not constructed yet.  You may need to run extract_postMetrics.py'
                              }

        with open(output_folder + unique_id + '.json', 'w', encoding='utf-8') as f:
            json.dump(results_dictionary, f, ensure_ascii=False, indent=4)

    """ search for the solution which maximizes satisfaction, and other which minimizes cost"""

    def init_subset_selection(self, solutions: List[Solution]) -> List[Solution]:

        return [], []

    def get_metrics_fields(self, result, repetition):
        """adds metrics of current repetition of the algorithm in the dictionary, for later insertion in json
        """
        metrics_fields: List[str] = []

        time = result["time"] if "time" in result else 'NaN'
        nds_update_time = result["nds_update_time"] if "nds_update_time" in result else 'NaN'
        sss_total_time = result["sss_total_time"] if "sss_total_time" in result else 'NaN'
        # ref point: nadir point + (nadir - best)/10 = 1 + (1-0)/10 = 1.1
        hv = metrics.calculate_hypervolume(result["population"], ref_x=1.1, ref_y=1.1)
        spread = metrics.calculate_spread(result["population"])
        numSolutions = metrics.calculate_numSolutions(result["population"])
        spacing = metrics.calculate_spacing(result["population"])

        self.metrics_dictionary['time'][repetition] = time
        self.metrics_dictionary['nds_update_time'][repetition] = nds_update_time
        self.metrics_dictionary['sss_total_time'][repetition] = sss_total_time
        self.metrics_dictionary['HV'][repetition] = hv
        self.metrics_dictionary['spread'][repetition] = spread
        self.metrics_dictionary['numSolutions'][repetition] = numSolutions
        self.metrics_dictionary['spacing'][repetition] = spacing

    def get_pareto(self, population) -> List:
        """converts cost and value of each individual in a pair of coordinates and
        stores them in a list of duples (x,y)
        """
        self.algorithm.reset()
        solution_points = []
        for sol in population:
            point = (sol.total_cost, sol.total_satisfaction)
            solution_points.insert(len(solution_points), point)
        return solution_points
