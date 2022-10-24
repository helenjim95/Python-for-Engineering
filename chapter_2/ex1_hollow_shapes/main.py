def print_triangle(n):
    k = 2
    for row in range(1, n + 1):
        for column in range(1, 2 * n):
            if row + column == n + 1 or column - row == n - 1:
                print("*", end="")
            elif row == n and column != k:
                print("*", end="")
                k = k + 2
            else:
                print(end=" ")
        print()

number = int(input("Enter half basewidth (total + 1) :")) + 1

print_triangle(number)