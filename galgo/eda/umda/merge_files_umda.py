# RUN SCRIPT IN THE FOLDER WHERE THE DATA IS LOCATED
import os

from algorithms.EDA.UMDA.umda_executer import UMDAExecuter
from algorithms.EDA.UMDA.umda_algorithm import UMDAAlgorithm

filenames = [f for f in os.listdir(
    os.getcwd()) if f.endswith('.txt') and "umda" in f]
print(filenames)

UMDAExecuter(UMDAAlgorithm).initialize_file(
    'output/metrics/merged_output_umda.txt')

# Open file in write mode
with open('output/metrics/merged_output_umda.txt', 'w') as outfile:
    # outfile.write("Dataset,Algorithm,Population Length,Generations,Evaluations,"
    #              "Selected Individuals,Time(s),AvgValue,BestAvgValue,BestGeneration,HV,Spread,NumSolutions,Spacing,"
    #              "NumGenerations,Requirements per sol,NumEvaluations\n")
    # Iterate through list
    for names in filenames:
        # Open each file in read mode
        with open(names) as infile:
            # read the data from file1 and
            # file2 and write it in file3
            outfile.write(infile.read())

        # Add '\n' to enter data of file2
        # from next line
        # outfile.write("\n")
    outfile.close()
