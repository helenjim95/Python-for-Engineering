import math
with open("OUTCAR.txt", "r") as readfile:
    iteration = 0
    CPU_list = []
    for line in readfile:
        line_strip = line.strip()
        if line_strip.startswith("LOOP:"):
            CPU_time = line_strip.split("cpu time  ")[1].split(":")[0]
            CPU_list.append(CPU_time)
            iteration += 1
            print("ITERATION " + str(iteration) + ":")
            print("== " + line_strip)
        elif line_strip.startswith("energy without entropy"):
            print("== " + line_strip)
        elif line_strip.__contains__("k-points in BZ"):
            kpoint = line_strip

    max_CPU = max(CPU_list)
    index = CPU_list.index(max_CPU) + 1
    print(f"Maximal cpu time in iteration {index}: {max_CPU}")
    print(kpoint)