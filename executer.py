def executer(algorithm, iterations=10):
	avg_time=0
	avg_hv=0
	avg_spread=0
	f = open("executer.txt", "a")
	for i in range(0,iterations):
		print("Executing iteration: ", i+1)
		result=algorithm.run()
		avg_time += result["time"]
		avg_hv += result["hv"]
		avg_spread += result["spread"]
		f.write("Time(s): ",result["time"],", HV: ",result["hv"],", Spread: ",result["spread"])

	f.write("\n")
	f.write("AVERAGE:")
	f.write("Average Time(s): ", avg_time/iterations, ", HV: ", avg_hv/iterations, ", Spread: ", avg_spread/iterations)


