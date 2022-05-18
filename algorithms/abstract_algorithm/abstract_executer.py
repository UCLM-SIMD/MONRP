import json
from abc import ABC
from typing import List

from algorithms.abstract_algorithm.abstract_algorithm import AbstractAlgorithm
import evaluation.metrics as metrics





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
            'HV': [None] * self.executions,
            'spread': [None] * self.executions,
            'numSolutions': [None] * self.executions,
            'spacing': [None] * self.executions,
            'mean_bits_per_sol': [None] * self.executions,
            'avgValue': [None] * self.executions,
            'bestAvgValue': [None] * self.executions,

        }

    def execute(self, executions: int, file_path: str) -> None:
        """Method that executes the algorithm a number of times and saves results in json global output file
        """
        paretos_list = [] # list of pareto lists
        for it in range(0, executions):
            self.algorithm.reset()
            result = self.algorithm.run()
            self.get_metrics_fields(result, it)
            pareto = self.get_pareto(result['population']) # get a list with pareto points
            paretos_list.insert(0, pareto)

        #  add/update results in json output file
        self.algorithm.config_dictionary['num_executions'] = executions
        unique_id = ''.join(str(c) for c in self.algorithm.config_dictionary.values())
        results_dictionary = {'parameters': self.algorithm.config_dictionary,
                              'metrics': self.metrics_dictionary,
                              'paretos': paretos_list}

        try:
            with open(file_path) as f:
                all_dictionaries = json.load(f)
                if unique_id in all_dictionaries:
                    all_dictionaries[unique_id].update(results_dictionary)
                else:
                    all_dictionaries[unique_id] = results_dictionary
        except IOError:  # first time so output file does not exist yet
            all_dictionaries = {unique_id: results_dictionary}

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(all_dictionaries, f, ensure_ascii=False, indent=4)


    def get_metrics_fields(self, result, repetition):
        """adds metrics of current repetition of the algorithm in the dictionary, for later insertion in json
        """
        metrics_fields: List[str] = []

        time = str(result["time"]) if "time" in result else 'NaN'
        hv = str(metrics.calculate_hypervolume(result["population"]))
        spread = str(metrics.calculate_spread(result["population"]))
        numSolutions = str(
            metrics.calculate_numSolutions(result["population"]))
        spacing = str(metrics.calculate_spacing(result["population"]))
        mean_bits_per_sol = str(
            metrics.calculate_mean_bits_per_sol(result["population"]))
        avgValue = str(metrics.calculate_avgValue(result["population"]))
        bestAvgValue = str(
            metrics.calculate_bestAvgValue(result["population"]))

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

    def file_write_line(self, file_path: str, line: str) -> None:
        """Aux method to write a line in a file
        """
        f = open(file_path, "a")
        f.write(line)
        f.close()

    def initialize_file(self, file_path: str) -> None:
        """Aux method to write the header of the file.
        """
        # add all fields
        fields = self.config_fields + self.metrics_fields
        header: str = self.get_string_from_fields(fields, end_line=True)
        file = open(file_path, "w")
        file.write(header)
        file.close()

    def reset_file(self, file_path: str) -> None:
        file = open(file_path, "w")
        file.close()

    def get_string_from_fields(self, fields_array: List[str], end_line: bool = True) -> str:
        """Aux method to generate a string line from a list of fields
        """
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

    def initialize_file_pareto(self, file_path: str) -> None:
        # print("Running...")
        f = open(file_path, "w")
        # f.write("Dataset,AlgorithmName,Cost,Value\n")
        f.close()

    def get_pareto(self, population) -> List:
        """converts cost and value of each individual in a pair of coordinates and
        stores them in a list of duples (x,y)
        """
        self.algorithm.reset()
        solution_points = []
        for sol in population:
           point=(sol.total_cost,sol.total_satisfaction)
           solution_points.insert(0,point)
        return solution_points



    """
    all_dictionaries is a dict of dicts.
    if id of results_dictionary already exists, values are overwritten in the corresponding dictionary in all_dictionaries.
    otherwise, it is inserted as a new dictionary
    """


