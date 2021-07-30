from algorithms.abstract_default.executer import Executer
import evaluation.metrics as metrics

class UMDAExecuter(Executer):
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.algorithm_type = "umda"

    def initialize_file(self, file_path):
        # print("Running...")
        f = open(file_path, "w")
        f.write("Dataset,Algorithm,Population Length,Generations,"#TODO
                "Selected Individuals,Time(s),AvgValue,BestAvgValue,BestGeneration,HV,Spread,NumSolutions,Spacing,NumGenerations\n")
        f.close()

    def reset_file(self, file_path):
        file = open(file_path, "w")
        file.close()

    def execute(self, executions, file_path):
        algorithm_name = self.algorithm.__class__.__name__
        dataset = self.algorithm.dataset_name
        population_length = self.algorithm.population_length
        generations = self.algorithm.max_generations
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

            f = open(file_path, "a")
            data = str(dataset) + "," + \
                str(algorithm_name) + "," + \
                str(population_length) + "," + \
                str(generations) + "," + \
                str(selected_individuals) + "," + \
                str(time) + "," + \
                str(avgValue) + "," + \
                str(bestAvgValue) + "," + \
                str(bestGeneration) + "," + \
                str(hv) + "," + \
                str(spread) + "," + \
                str(numSolutions) + "," + \
                str(spacing) + "," + \
                str(numGenerations) + \
                "\n"

            f.write(data)
            f.close()

