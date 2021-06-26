from abc import ABC

class Executer(ABC):
    def __init__(self):
        pass

    def initialize_file(self,file_path):
        pass

    def reset_file(self,file_path):
        pass

    def execute(self, executions, file_path):
        pass

    
    def initialize_file_pareto(self, file_path):
        # print("Running...")
        f = open(file_path, "w")
        #f.write("Dataset,AlgorithmName,Cost,Value\n")
        f.close()

    def execute_pareto(self, file_path):
        algorithm_name = self.algorithm.get_name()
        dataset = self.algorithm.dataset_name

        result = self.algorithm.run()
        for sol in result["population"]:
            #print("Executing iteration: ", i + 1)
            cost = sol.objectives[1].value
            value = sol.objectives[0].value
            f = open(file_path, "a")
            data = str(cost) + "," + \
                str(value) + \
                "\n"

            f.write(data)
            f.close()