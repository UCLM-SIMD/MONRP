from algorithms.abstract_algorithm.abstract_executer import AbstractExecuter
import evaluation.metrics as metrics

class UMDAExecuter(AbstractExecuter):
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.algorithm_type = "umda"

    def initialize_file(self, file_path):
        # print("Running...")
        f = open(file_path, "w")
        f.write("Dataset,Algorithm,Population Length,Generations,Evaluations,"
                "Selected Individuals,Time(s),AvgValue,BestAvgValue,BestGeneration,HV,Spread,NumSolutions,Spacing,"
                "NumGenerations,Requirements per sol\n")
        f.close()

    def reset_file(self, file_path):
        file = open(file_path, "w")
        file.close()

    def execute(self, executions, file_path):
        algorithm_name = self.algorithm.__class__.__name__
        dataset_name = self.algorithm.dataset_name
        population_length = self.algorithm.population_length
        generations = self.algorithm.max_generations
        evaluations = self.algorithm.max_evaluations
        selected_individuals = self.algorithm.selected_individuals
        dataset = self.algorithm.dataset

        for i in range(0, executions):
            result = self.algorithm.run()

            time = str(result["time"]) if "time" in result else 'NaN'
            numGenerations = str(
                result["numGenerations"]) if "numGenerations" in result else 'NaN'
            bestGeneration = str(
                result["bestGeneration"]) if "bestGeneration" in result else 'NaN' #TODO no hay bestgen

            avgValue = str(metrics.calculate_avgValue(result["population"]))
            bestAvgValue = str(
                metrics.calculate_bestAvgValue(result["population"]))
            hv = str(metrics.calculate_hypervolume(result["population"]))
            spread = str(metrics.calculate_spread(
                result["population"], dataset))
            numSolutions = str(
                metrics.calculate_numSolutions(result["population"]))
            spacing = str(metrics.calculate_spacing(result["population"]))
            mean_bits_per_sol =  str(metrics.calculate_mean_bits_per_sol(result["population"]))
            numEvaluations = str(
                result["numEvaluations"]) if "numEvaluations" in result else 'NaN'

            f = open(file_path, "a")
            data = str(dataset_name) + "," + \
                str(algorithm_name) + "," + \
                str(population_length) + "," + \
                str(generations) + "," + \
                str(evaluations) + "," + \
                str(selected_individuals) + "," + \
                str(time) + "," + \
                str(avgValue) + "," + \
                str(bestAvgValue) + "," + \
                str(bestGeneration) + "," + \
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

