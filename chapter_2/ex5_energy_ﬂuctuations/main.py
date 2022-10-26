import math
import matplotlib as mpl
import matplotlib.pyplot as plt

with open("average.csv", "w") as output:
    with open("zif-nvt.csv", "r") as file:
        ps_list = []
        values = []
        avg_values = []
        # ps, value = file.readline().lstrip().replace("\n", "").split(sep="  ")
        # ps_list.append(0)
        # values.append(float(value))
        for line in file:
            ps, value = line.lstrip().split(sep="  ")
            ps_list.append(int(float(ps)))
            values.append(float(value))

        # calculate average value
        # datapoint 1 (average 1-24) datapoint 2 (average 2-25), datapoint (3-26)
        total_list = []
        WINDOW_SIZE = 25
        # for 24 iterations, add from values[0]...values[24] and average
        for i in range(len(values) - WINDOW_SIZE + 1):
            total = 0
            for index in range(WINDOW_SIZE):
                total = values[i + index] + total
            avg_value = total / WINDOW_SIZE
            avg_values.append(avg_value)

        for i in range(len(avg_values)):
            data_list = [str(ps_list[i]), str("{:.2f}".format(values[i])), str("{:.2f}".format(avg_values[i]))]
            print(data_list)
            output.write(" ".join(data_list))
            output.write("\n")

        # plot the data
        plt.plot(range(len(values)), values)
        plt.plot(range(len(avg_values)), avg_values)
        plt.show()
        plt.savefig("plot.pdf")
