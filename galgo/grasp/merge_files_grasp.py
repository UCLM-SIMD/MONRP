# RUN SCRIPT IN THE FOLDER WHERE THE DATA IS LOCATED
import os

from algorithms.GRASP.grasp_executer import GRASPExecuter
from algorithms.GRASP.GRASP import GRASP

filenames = [f for f in os.listdir(
    os.getcwd()) if f.endswith('.txt') and "grasp" in f]
print(filenames)

GRASPExecuter(GRASP).initialize_file('output/metrics/merged_output_grasp.txt')

# Open file3 in write mode
with open('output/metrics/merged_output_grasp.txt', 'w') as outfile:
    # outfile.write("Dataset,Algorithm,Iterations,Solutions per Iteration,Evaluations,Initialization Type,"
    #            "Local Search Type,Path Relinking,Time(s),AvgValue,BestAvgValue,HV,Spread,NumSolutions,"
    #			"Spacing,NumGenerations,Requirements per sol,NumEvaluations\n")
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
