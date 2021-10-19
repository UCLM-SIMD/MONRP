import evaluation.metrics as metrics
from algorithms.abstract_default.executer import Executer


class GRASPExecuter(Executer):
    def __init__(self, algorithm):
        self.algorithm_type = "grasp"
        self.algorithm = algorithm

    def initialize_file(self, file_path):
        # print("Running...")
        f = open(file_path, "w")
        f.write("Dataset,Algorithm,Iterations,Solutions per Iteration,Evaluations,Initialization Type"
                "Local Search Type,Path Relinking,Time(s),AvgValue,BestAvgValue,HV,Spread,NumSolutions,Spacing,"
                "Requirements per sol,NumEvaluations\n")
        f.close()

    def reset_file(self, file_path):
        file = open(file_path, "w")
        file.close()

    def execute(self, executions, file_path):
        algorithm_name = self.algorithm.__class__.__name__
        dataset_name = self.algorithm.dataset_name
        iterations = self.algorithm.iterations
        solutions_per_iteration = self.algorithm.solutions_per_iteration
        evaluations = self.algorithm.max_evaluations
        local_search_type = self.algorithm.local_search_type
        init_type = self.algorithm.init_type
        path_relinking = self.algorithm.path_relinking_mode
        dataset = self.algorithm.dataset

        for i in range(0, executions):
            #print("Executing iteration: ", i + 1)
            self.algorithm.reset()
            result = self.algorithm.run()

            #print(result)

            time = str(result["time"]) if "time" in result else 'NaN'
            numGenerations = str(
                result["numGenerations"]) if "numGenerations" in result else 'NaN'

            avgValue = str(metrics.calculate_avgValue(result["population"]))
            bestAvgValue = str(
                metrics.calculate_bestAvgValue(result["population"]))
            hv = str(metrics.calculate_hypervolume(result["population"]))
            spread = str(metrics.calculate_spread(
                result["population"], dataset))
            numSolutions = str(
                metrics.calculate_numSolutions(result["population"]))
            spacing = str(metrics.calculate_spacing(result["population"]))
            mean_bits_per_sol = str(
                metrics.calculate_mean_bits_per_sol(result["population"]))
            numEvaluations = str(
                result["numEvaluations"]) if "numEvaluations" in result else 'NaN'

            f = open(file_path, "a")
            data = str(dataset_name) + "," + \
                str(algorithm_name) + "," + \
                str(iterations) + "," + \
                str(solutions_per_iteration) + "," + \
                str(evaluations) + "," + \
                str(init_type) + "," + \
                str(local_search_type) + "," + \
                str(path_relinking) + "," + \
                str(time) + "," + \
                str(avgValue) + "," + \
                str(bestAvgValue) + "," + \
                str(hv) + "," + \
                str(spread) + "," + \
                str(numSolutions) + "," + \
                str(spacing) + "," + \
                str(numGenerations) + "," + \
                str(mean_bits_per_sol) + "," + \
                str(numEvaluations) + \
                "\n"

            f.write(data)
            f.close()

        # print("End")
