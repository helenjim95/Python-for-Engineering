import math
import matplotlib as mpl
import matplotlib.pyplot as plt

with open("average.csv", "w") as output:
    with open("__files/zif-nvt.csv", "r") as file:
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
        total_list = []
        WINDOW_SIZE = 25
        # for first 24 iterations, average what you got
        for i in range(len(values) - WINDOW_SIZE + 1):
            total = 0
            count = 0
            if i < WINDOW_SIZE - 1:
                for index in range(i + 1):
                    # print(f"index: {index}")
                    # print(f"values[{index}] {values[index]} + {total}")
                    total += values[index]
                    count += 1
                avg_value = total / count
                # print(f"total: {total}")
                # print(f"average: {avg_value}")
            else:
                # starting from 25th iteration, average 25 datapoints
                # print(f"i: {i}")
                for index in range(WINDOW_SIZE):
                    # print(f"index: {index}")
                    # print(f"values[{i + index}] {values[i + index]} + {total}")
                    total = values[i + index] + total
                # print(f"total: {total}")
                avg_value = total / WINDOW_SIZE
            # print(f"average: {avg_value}")
            avg_values.append(avg_value)

        # format to output
        for i in range(len(avg_values)):
            data_list = str("{:.2f}".format(avg_values[i]))
            # print(data_list)
            output.write(data_list + "\n")

        # plot the data
        plt.plot(range(len(values)), values)
        plt.plot(range(len(avg_values)), avg_values)
        # plt.show()
        plt.savefig("plot.pdf")
