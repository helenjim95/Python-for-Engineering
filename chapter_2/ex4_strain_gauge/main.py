import math

with open("strain_gauge_processed.csv", "w") as output:
    with open("__files/strain_gauge_rosette.csv", "r") as file:
        r2_1_list = []
        r2_2_list = []
        r2_3_list = []
        e_1_list = []
        e_2_list = []
        first_line = file.readline().replace("\n", "").split(sep="	")
        first_line.append("e_1")
        first_line.append("e_2")
        output.write(";".join(first_line))
        output.write("\n")
        for line in file:
            data = line.split(sep="	")
            # need to change decimal to 9
            r2_1 = float(data[0])
            r2_2 = float(data[1])
            r2_3 = float(data[2])
            r2_1_list.append(r2_1)
            r2_2_list.append(r2_2)
            r2_3_list.append(r2_3)
            e_1 = 0.5 * (r2_1 + r2_3 + math.sqrt(2 * ((r2_1 - r2_2) ** 2 + (r2_2 - r2_3) ** 2)))
            e_2 = 0.5 * (r2_1 + r2_3 - math.sqrt(2 * ((r2_1 - r2_2) ** 2 + (r2_2 - r2_3) ** 2)))
            e_1_list.append(e_1)
            e_2_list.append(e_2)

            data_list = [str("{:.9f}".format(float(r2_1))), str("{:.9f}".format(float(r2_2))), str("{:.9f}".format(float(r2_3))), str("{:.9f}".format(float(e_1))), str("{:.9f}".format(float(e_2)))]
            output.write(";".join(data_list))
            output.write("\n")
