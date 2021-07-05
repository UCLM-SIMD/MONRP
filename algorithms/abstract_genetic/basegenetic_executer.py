import evaluation.metrics as metrics

from algorithms.abstract_default.executer import Executer


class BaseGeneticExecuter(Executer):
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.algorithm_type = "genetic"

    def initialize_file(self, file_path):
        # print("Running...")
        f = open(file_path, "w")
        f.write("Dataset,Algorithm,Population Length,Generations,"
                "Selection Scheme,Selection Candidates,Crossover Scheme,Crossover Probability,Mutation Scheme,"
                "Mutation Probability,Replacement Scheme,Time(s),AvgValue,BestAvgValue,BestGeneration,HV,Spread,NumSolutions,Spacing,NumGenerations\n")
        f.close()

    def reset_file(self, file_path):
        file = open(file_path, "w")
        file.close()

    def execute(self, executions, file_path):
        algorithm_name = self.algorithm.__class__.__name__
        dataset = self.algorithm.dataset_name
        population_length = self.algorithm.population_length
        generations = self.algorithm.max_generations
        selection = self.algorithm.selection_scheme
        selection_candidates = self.algorithm.selection_candidates
        crossover = self.algorithm.crossover_scheme
        crossover_prob = self.algorithm.crossover_prob
        mutation = self.algorithm.mutation_scheme
        mutation_prob = self.algorithm.mutation_prob
        replacement = self.algorithm.replacement_scheme
        dataset = self.algorithm.dataset

        for i in range(0, executions):
            #print("Executing iteration: ", i + 1)
            result = self.algorithm.run()

            time = str(result["time"]) if "time" in result else 'NaN'
            numGenerations = str(
                result["numGenerations"]) if "numGenerations" in result else 'NaN'
            bestGeneration = str(
                result["bestGeneration"]) if "bestGeneration" in result else 'NaN'

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
                str(selection) + "," + \
                str(selection_candidates) + "," + \
                str(crossover) + "," + \
                str(crossover_prob) + "," + \
                str(mutation) + "," + \
                str(mutation_prob) + "," + \
                str(replacement) + "," + \
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

        # print("End")
