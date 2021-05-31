from evaluation.metrics import *
from algorithms.abstract.executer import Executer


class NSGAIIExecuter(Executer):
    def __init__(self):
        self.algorithm_type = "nsgaii"

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

    def execute(self, algorithm, dataset, executions, file_path):
        algorithm_name = algorithm.__class__.__name__
        dataset = dataset
        population_length = algorithm.population_length
        generations = algorithm.max_generations
        selection = algorithm.selection_scheme
        selection_candidates = algorithm.selection_candidates
        crossover = algorithm.crossover_scheme
        crossover_prob = algorithm.crossover_prob
        mutation = algorithm.mutation_scheme
        mutation_prob = algorithm.mutation_prob
        replacement = algorithm.replacement_scheme

        for i in range(0, executions):
            #print("Executing iteration: ", i + 1)
            result = algorithm.run()

            time = str(result["time"]) if "time" in result else 'NaN'
            numGenerations = str(
                result["numGenerations"]) if "numGenerations" in result else 'NaN'
            bestGeneration = str(
                result["bestGeneration"]) if "bestGeneration" in result else 'NaN'

            avgValue = str(calculate_avgValue(result["population"]))
            bestAvgValue = str(calculate_bestAvgValue(result["population"]))
            hv = str(calculate_hypervolume(result["population"]))
            spread = str(calculate_spread(result["population"]))
            numSolutions = str(calculate_numSolutions(result["population"]))
            spacing = str(calculate_spacing(result["population"]))

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
