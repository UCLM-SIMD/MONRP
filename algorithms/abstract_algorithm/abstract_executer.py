from abc import ABC, abstractmethod
from typing import List

from algorithms.abstract_algorithm.abstract_algorithm import AbstractAlgorithm
import evaluation.metrics as metrics


class AbstractExecuter(ABC):
    """Executer class used to delegate configuration, execution and formatting of algorithm outputs
    """

    def __init__(self, algorithm: AbstractAlgorithm):
        """All executers store default config and metrics fields. Specific implementations might include more fields
        """
        self.algorithm: AbstractAlgorithm = algorithm
        self.config_fields: List[str] = ["Dataset", "Algorithm"]
        self.metrics_fields: List[str] = ["Time(s)", "HV", "Spread", "NumSolutions", "Spacing",
                                          "Requirements per sol", "AvgValue", "BestAvgValue", ]

    def execute(self, executions: int, file_path: str) -> None:
        """Method that executes the algorithm a number of times and writes a new line with config and metrics data for each execution

        Args:
            executions (int): [description]
            file_path (str): [description]
        """
        config_fields: List[str] = self.get_config_fields()
        for i in range(0, executions):
            #print("Executing iteration: ", i + 1)
            self.algorithm.reset()
            result = self.algorithm.run()
            metrics_fields: List[str] = self.get_metrics_fields(result)

            config_line = self.get_string_from_fields(
                config_fields, end_line=False)
            metrics_line = self.get_string_from_fields(
                metrics_fields, end_line=True)
            line = config_line+metrics_line
            self.file_write_line(file_path, line)

        # print("End")

    def get_config_fields(self,) -> List[str]:
        """Method that returns the config fields of the algorithm. Specific implementations might add new fields.
        """
        config_lines: List[str] = []

        algorithm_name = self.algorithm.__class__.__name__
        dataset_name = self.algorithm.dataset_name

        config_lines.append(str(dataset_name))
        config_lines.append(str(algorithm_name))

        return config_lines

    def get_metrics_fields(self, result) -> List[str]:
        """Method that returns the metrics fields of an execution result. Specific implementations might add new fields.
        """
        metrics_fields: List[str] = []

        time = str(result["time"]) if "time" in result else 'NaN'
        hv = str(metrics.calculate_hypervolume(result["population"]))
        spread = str(metrics.calculate_spread(
            result["population"], self.algorithm.dataset))
        numSolutions = str(
            metrics.calculate_numSolutions(result["population"]))
        spacing = str(metrics.calculate_spacing(result["population"]))
        mean_bits_per_sol = str(
            metrics.calculate_mean_bits_per_sol(result["population"]))
        avgValue = str(metrics.calculate_avgValue(result["population"]))
        bestAvgValue = str(
            metrics.calculate_bestAvgValue(result["population"]))

        metrics_fields.append(str(time))
        metrics_fields.append(str(hv))
        metrics_fields.append(str(spread))
        metrics_fields.append(str(numSolutions))
        metrics_fields.append(str(spacing))
        metrics_fields.append(str(mean_bits_per_sol))
        metrics_fields.append(str(avgValue))
        metrics_fields.append(str(bestAvgValue))

        return metrics_fields

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

        self.file_write_line(file_path, header)

    def reset_file(self, file_path: str) -> None:
        file = open(file_path, "w")
        file.close()

    def get_string_from_fields(self, fields_array: List[str], end_line: bool = True) -> str:
        """Aux method to generate a string line from a list of fields
        """
        line: str = ""
        for field_index in range(len(fields_array)-1):
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

    def execute_pareto(self, file_path: str) -> None:
        """Method that executes the algorithm once and writes the solution points.
        """
        self.algorithm.reset()
        result = self.algorithm.run()
        for sol in result["population"]:
            #print("Executing iteration: ", i + 1)
            cost = sol.total_cost
            value = sol.total_satisfaction

            f = open(file_path, "a")
            data = f"{str(cost)},{str(value)}\n"

            f.write(data)
            f.close()
