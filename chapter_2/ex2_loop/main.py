import math
for i in range(0, 316, 5):
    number = i/100
    result = math.sin(number)
    if result < 0.1:
        print("sin({:.2f}) = {:.2f}, near null".format(number, result))
    else:
        print("sin({:.2f}) = {:.2f}".format(number, result))