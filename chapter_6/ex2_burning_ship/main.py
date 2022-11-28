from __future__ import division
from PIL import Image


def burning_ship(z, complex_number):
    return complex(abs(z.real), abs(z.imag)) * complex(abs(z.real), abs(z.imag)) + complex_number


def calc_fractal_point(complex_number, burning_ship, maximum_iterations):
    """Calculate the fractal point."""
    z = 0
    VECTOR = 2
    for iteration in range(maximum_iterations):
        z = burning_ship(z, complex_number)
        if abs(z) > VECTOR:
            return iteration
    return maximum_iterations


def convert_pixel_coords_to_viewport_complex_number(xycoords, image_size, view_center, view_size):
    return complex((xycoords[0] / image_size[0] - 0.5) * view_size[0] + view_center[0],
                   (xycoords[1] / image_size[1] - 0.5) * view_size[1] + view_center[1])

# line
# for each pixel(x, y) on the screen, do:
#     x := scaled x coordinate of pixel(scaled to lie in the Mandelbrot X scale(-2.5, 1))
#     y := scaled y coordinate of pixel(scaled to lie in the Mandelbrot Y scale(-1, 1))
#
#     zx := x # zx represents the real part of z
#     zy := y # zy represents the imaginary part of z
#     iteration = 0
#     max_iteration = 100
#
#     while (zx * zx + zy * zy < 4) and (iteration < max_iteration):
#         xtemp = zx * zx - zy * zy + x
#         zy = abs(2 * zx * zy) + y # abs returns the absolute value
#         zx = xtemp
#         iteration += 1
#
#     if iteration == max_iteration:
#         return insideColor
#     else:
#         return iteration * color


if __name__ == "__main__":
    MAXIMUM_ITERATION = 100
    IMAGE_SIZE = (800, 450)
    VIEW_CENTER = (-0.5, 0.0)
    VIEW_SIZE = (3.0, 3.0)
    MAX_COLOR = 255

    image_mode = 'I'
    img = Image.new(image_mode, IMAGE_SIZE)

    for y in range(IMAGE_SIZE[1]):
        for x in range(IMAGE_SIZE[0]):
            complex_number = convert_pixel_coords_to_viewport_complex_number((x, y), IMAGE_SIZE, VIEW_CENTER, VIEW_SIZE)
            iteration = calc_fractal_point(complex_number, burning_ship, MAXIMUM_ITERATION)
            if iteration == MAXIMUM_ITERATION:
                color = 0
            else:
                color = int(iteration / MAXIMUM_ITERATION * MAX_COLOR)
            img.putpixel((x, y), color)
    img.save('fractal.png')
    img.show()

