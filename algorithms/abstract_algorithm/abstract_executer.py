import json
import warnings
from abc import ABC
from typing import List

from algorithms.abstract_algorithm.abstract_algorithm import AbstractAlgorithm
import evaluation.metrics as metrics
from models.Solution import Solution


class AbstractExecuter(ABC):
    """Executer class used to delegate configuration, execution and formatting of algorithm outputs
    """

    def __init__(self, algorithm: AbstractAlgorithm, excecs: int):
        """All executers store default config and metrics fields. Specific implementations might include more fields
        """
        self.executions = int(excecs)
        self.algorithm: AbstractAlgorithm = algorithm
        self.config_fields: List[str] = ["Dataset", "Algorithm"]
        self.metrics_fields: List[str] = ["Time(s)", "HV", "Spread", "NumSolutions", "Spacing",
                                          "Requirements per sol", "AvgValue", "BestAvgValue", ]

        self.metrics_dictionary = {
            'time': [None] * self.executions,
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

            self.metrics_dictionary['NDS_size'][it] = len(result['population']) # store original NDS size created in search
            result['population'] = self.search_solution_subset(result['population'])

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

    """ finds the subset which maximizes HV with self.subset_size of solutions.
    The search is a basic/tradicional greedy forward search based on the HV metric
    A fixed reference point is assumed during the search (always the same), as in 
    'Greedy Hypervolume Subset Selection in Low Dimensions,Evolutionary Computation 24(3): 521-544'
    where fixed r is upper bounds (in minimizatino problems), that is, the nadir point
    
    """

    def search_solution_subset(self, solutions: [Solution]) -> [Solution]:

        if len(solutions) < self.algorithm.subset_size:

            warnings.warn('|solutions| < subset_size parameter!! Solution subset set to original final solution', UserWarning)
            return solutions

        #ref_x, ref_y = metrics.find_ref_points(solutions) # need ref points taking all solutions into account
        indices_selected = []
        subset = []
        for _ in range(0, self.algorithm.subset_size):
            best_hv = -1
            best_index = -1
            for i in range(0, len(solutions)):
                if not i in indices_selected:
                    subset.insert(len(subset), solutions[i])
                    hv = metrics.calculate_hypervolume(subset, ref_x=1, ref_y=1)
                    if hv > best_hv:
                        best_hv = hv
                        best_index = i
                    del subset[-1]
            if best_index != -1:
                subset.insert(len(subset), solutions[best_index])
                indices_selected.insert(len(indices_selected), best_index)

        return subset

    """ search for the solution which maximizes satisfaction, and other which minimizes cost"""

    def init_subset_selection(self, solutions: [Solution]) -> [Solution]:

        return [], []

    def get_metrics_fields(self, result, repetition):
        """adds metrics of current repetition of the algorithm in the dictionary, for later insertion in json
        """
        metrics_fields: List[str] = []

        time = result["time"] if "time" in result else 'NaN'
        # worst possible values as ref points (solutions are scaled)
        hv = metrics.calculate_hypervolume(result["population"], ref_x=1, ref_y=1)
        spread = metrics.calculate_spread(result["population"])
        numSolutions = metrics.calculate_numSolutions(result["population"])
        spacing = metrics.calculate_spacing(result["population"])
        mean_bits_per_sol = metrics.calculate_mean_bits_per_sol(result["population"])
        avgValue = metrics.calculate_avgValue(result["population"])
        bestAvgValue = metrics.calculate_bestAvgValue(result["population"])

        self.metrics_dictionary['time'][repetition] = time
        self.metrics_dictionary['HV'][repetition] = hv
        self.metrics_dictionary['spread'][repetition] = spread
        self.metrics_dictionary['numSolutions'][repetition] = numSolutions
        self.metrics_dictionary['spacing'][repetition] = spacing
        self.metrics_dictionary['mean_bits_per_sol'][repetition] = mean_bits_per_sol
        self.metrics_dictionary['avgValue'][repetition] = avgValue
        self.metrics_dictionary['bestAvgValue'][repetition] = bestAvgValue

        # metrics_fields.append(str(time))
        # metrics_fields.append(str(hv))
        # metrics_fields.append(str(spread))
        # metrics_fields.append(str(numSolutions))
        # metrics_fields.append(str(spacing))
        # metrics_fields.append(str(mean_bits_per_sol))
        # metrics_fields.append(str(avgValue))
        # metrics_fields.append(str(bestAvgValue))

        # return metrics_fields

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


"""
    def file_write_line(self, file_path: str, line: str) -> None:
        #Aux method to write a line in a file
        
        f = open(file_path, "a")
        f.write(line)
        f.close()
"""

""" def initialize_file(self, file_path: str) -> None:
        #Aux method to write the header of the file.
        
        # add all fields
        fields = self.config_fields + self.metrics_fields
        header: str = self.get_string_from_fields(fields, end_line=True)
        file = open(file_path, "w")
        file.write(header)
        file.close()
"""
"""   def reset_file(self, file_path: str) -> None:
        file = open(file_path, "w")
        file.close()
"""
"""   def get_string_from_fields(self, fields_array: List[str], end_line: bool = True) -> str:
        #Aux method to generate a string line from a list of fields
        
        line: str = ""
        for field_index in range(len(fields_array) - 1):
            line += f"{fields_array[field_index]},"
        line += f"{fields_array[-1]}"

        # add end of line to last field or comma
        if end_line:
            line += "\n"
        else:
            line += ","
        return line
"""
"""   def initialize_file_pareto(self, file_path: str) -> None:
        # print("Running...")
        f = open(file_path, "w")
        # f.write("Dataset,AlgorithmName,Cost,Value\n")
        f.close()
"""
