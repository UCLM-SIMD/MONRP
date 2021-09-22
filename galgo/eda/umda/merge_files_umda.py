# RUN SCRIPT IN THE FOLDER WHERE THE DATA IS LOCATED
import os

filenames = [f for f in os.listdir(os.getcwd()) if f.endswith('.txt') and "umda" in f]
print(filenames)

# Open file in write mode
with open('merged_output_grasp.txt', 'w') as outfile:
	outfile.write("Dataset,Algorithm,Population Length,Generations,Evaluations,"
                "Selected Individuals,Time(s),AvgValue,BestAvgValue,BestGeneration,HV,Spread,NumSolutions,Spacing,"
                "NumGenerations,Requirements per sol\n")
	# Iterate through list
	for names in filenames:
		# Open each file in read mode
		with open(names) as infile:
			# read the data from file1 and
			# file2 and write it in file3
			outfile.write(infile.read())

		# Add '\n' to enter data of file2
		# from next line
		#outfile.write("\n")
	outfile.close()