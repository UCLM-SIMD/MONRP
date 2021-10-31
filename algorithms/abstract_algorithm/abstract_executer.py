from abc import ABC, abstractmethod


class AbstractExecuter(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize_file(self, file_path) -> None:
        pass

    @abstractmethod
    def reset_file(self, file_path) -> None:
        pass

    @abstractmethod
    def execute(self, executions, file_path) -> None:
        pass

    def initialize_file_pareto(self, file_path) -> None:
        # print("Running...")
        f = open(file_path, "w")
        # f.write("Dataset,AlgorithmName,Cost,Value\n")
        f.close()

    def execute_pareto(self, file_path) -> None:
        algorithm_name = self.algorithm.get_name()
        dataset = self.algorithm.dataset_name

        result = self.algorithm.run()
        for sol in result["population"]:
            #print("Executing iteration: ", i + 1)
            cost = sol.total_cost
            value = sol.total_satisfaction
            f = open(file_path, "a")
            data = str(cost) + "," + \
                str(value) + \
                "\n"

            f.write(data)
            f.close()
