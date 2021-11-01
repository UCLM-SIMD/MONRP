# RUN SCRIPT IN THE FOLDER WHERE THE DATA IS LOCATED
import os

from algorithms.EDA.PBIL.pbil_executer import PBILExecuter

filenames = [f for f in os.listdir(
    os.getcwd()) if f.endswith('.txt') and "pbil" in f]
print(filenames)

PBILExecuter().initialize_file('output/metrics/merged_output_pbil.txt')

# Open file3 in write mode
with open('merged_output_pbil.txt', 'w') as outfile:
    # outfile.write("Dataset,Algorithm,Population Length,Generations,Evaluations,"
    #            "Learning Rate,Mutation Probability,Mutation Shift,Time(s),AvgValue,BestAvgValue,BestGeneration,HV,Spread,NumSolutions,Spacing,"
    #            "NumGenerations,Requirements per sol,NumEvaluations\n")
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
