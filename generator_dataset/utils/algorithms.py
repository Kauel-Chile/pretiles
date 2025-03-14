import numpy as np
import cv2

def bresenham_line(start=(10, 10), end=(10, 60), gsd=0.25):
    """
    Generate a list of points using Bresenham's Line Algorithm.

    This algorithm computes the points between two coordinates (start, end),
    producing a list of tuples representing the line connecting these two points
    based on the principle of rasterizing a line.

    Parameters:
        start (tuple): The starting point of the line as (x1, y1).
        end (tuple): The ending point of the line as (x2, y2).

    Returns:
        list: A list of points (x, y) between the start and end points.

    Examples:
        >>> points1 = generate_bresenham_line((0, 0), (3, 4))
        >>> points2 = generate_bresenham_line((3, 4), (0, 0))
        >>> assert(set(points1) == set(points2))
        >>> print(points1)
        [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
        >>> print(points2)
        [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """

    # Unpack start and end points
    x1, y1 = np.array(start, dtype=np.int32).copy()
    x2, y2 = np.array(end, dtype=np.int32).copy()


    # Calculate differences
    dx = x2 - x1
    dy = y2 - y1

    # Determine if the line is steep (more vertical than horizontal)
    is_steep = abs(dy) > abs(dx)

    # Swap coordinates if the line is steep (for easier handling)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Ensure the line is always drawn left-to-right
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    # Recalculate differences after the possible swap
    dx = x2 - x1
    dy = y2 - y1

    # Error term initialized to half of dx
    error = dx // 2
    ystep = 1 if y1 < y2 else -1  # Determines whether to increment or decrement y

    # List to store the generated points
    points = []
    y = y1

    # Main loop for Bresenham's algorithm
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)  # Swap x and y if the line is steep
        points.append(coord)

        # Update error term
        error -= abs(dy)

        # If error is negative, adjust y and reset the error term
        if error < 0:
            y += ystep
            error += dx

    return points

def generate_parallel_or_perpendicular_lines(image_shape, num_lines=2):
    """
    Generate a specified number of parallel or perpendicular lines within an image.

    Parameters:
    image_shape (tuple): Shape of the image (height, width).
    num_lines (int): Number of lines to generate.

    Returns:
    list: A list of lines, where each line is represented as a list of points.
    """
    lines = []
    lens = []
    for _ in range(num_lines):
        # Choose a random orientation (horizontal, vertical, or diagonal)
        orientation = np.random.choice(['horizontal', 'vertical', 'diagonal'])

        # Generate pseudo-random start and end points
        if orientation == 'horizontal':
            y = np.random.randint(0, image_shape[0])  # Fix Y, random in X
            x = np.random.randint(0, image_shape[1] // 2)
            x2 = x + np.random.randint(0, image_shape[1] - x)
            start = (x, y)
            end = (x2, y)
        elif orientation == 'vertical':
            x = np.random.randint(0, image_shape[1])  # Fix X, random in Y
            y = np.random.randint(0, image_shape[0] // 2)
            y2 = y + np.random.randint(0, image_shape[0] - y)
            start = (x, y)
            end = (x, y2)
        else:  # Diagonal
            x = np.random.randint(0, image_shape[1] // 2)
            x2 = x + np.random.randint(0, image_shape[1] - x)
            y = np.random.randint(0, image_shape[0] // 2)
            y2 = y + np.random.randint(0, image_shape[0] - y)
            start = (x, y)
            end = (x2, y2)

        # Add the generated line
        line = bresenham_line(start=start, end=end)
        lines.append(line)
        x1, y1 = start
        x2, y2 = end
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        lens.append(length)

    return lines, lens

# def generate_parallel_lines_adjacent(image_shape):
#     """
#     Generates parallel lines of a specified orientation in an image.

#     Parameters:
#     image_shape (tuple): The shape of the image (height, width).

#     Returns:
#     list: A list of lists, where each inner list represents a parallel line.
#     """
#     lines = []
#     orientation = np.random.choice(['horizontal', 'vertical', 'diagonal'])
#     distance = 50  # Distance between parallel lines

#     # Random starting points for different orientations
#     x = np.random.randint(0, image_shape[1])
#     y = np.random.randint(0, image_shape[0])
#     x_start = np.random.randint(0, image_shape[1] // 2)
#     y_start = np.random.randint(0, image_shape[0] // 2)

#     for i in range(3):
#         if orientation == 'horizontal':
#             y_offset = i * distance
#             start = (0, y + y_offset)
#             end = (image_shape[1], y + y_offset)
#         elif orientation == 'vertical':
#             x_offset = i * distance
#             start = (x + x_offset, 0)
#             end = (x + x_offset, image_shape[0])
#         else:  # Diagonal
#             x_offset = i * distance
#             y_offset = i * distance
#             start = (x_start + x_offset, y_start + y_offset)
#             end = (image_shape[1] - x_start + x_offset, image_shape[0] - y_start + y_offset)

#         start = (max(0, min(start[0], image_shape[1] - 1)), max(0, min(start[1], image_shape[0] - 1)))
#         end = (max(0, min(end[0], image_shape[1] - 1)), max(0, min(end[1], image_shape[0] - 1)))


#         # Generate the line using Bresenham's line algorithm
#         parallel_line = bresenham_line(start, end)
#         if parallel_line:
#             lines.append(parallel_line)

#     return lines

def generate_parallel_lines_adjacent(image_shape, gsd):
    lines = []
    lens = []
    x = np.random.randint(0, image_shape[1] // 2)
    x2 = x + np.random.randint(0, image_shape[1] - x)
    y = np.random.randint(0, image_shape[0] // 2)
    y2 = y + np.random.randint(0, image_shape[0] - y)
    start = (x, y)
    end = (x2, y2)

    line_direction = np.array(end, dtype=np.float32) - np.array(start, dtype=np.float32) 
    line_direction_norm = line_direction / np.linalg.norm(line_direction) 
    line_per = np.array((line_direction_norm[1], -line_direction_norm[0]))
    line_per = line_per / np.linalg.norm(line_per)

    for i in range(3):
        if i == 2 :
            p1 = start + line_per * i * (2.2 / gsd)
        else: 
            p1 = start + line_per * i * (1.4 / gsd)

        p2 = p1 + line_direction
        p1[0] = np.clip(p1[0], 0, image_shape[0] - 1)
        p1[1] = np.clip(p1[1], 0, image_shape[1] - 1)
        p2[0] = np.clip(p2[0], 0, image_shape[0] - 1)
        p2[1] = np.clip(p2[1], 0, image_shape[1] - 1)

        # Generate the line using Bresenham's line algorithm
        parallel_line = list(bresenham_line(p1, p2))
        lines.append(parallel_line)
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        lens.append(length)


    return lines, lens


def perlin_noise(shape=(512,512,3), min_inc=-1.0, max_exc=1.0, octaves=(0.25,0.25,0.25,0.25), dtype=np.float32):
    """
    Generates Perlin noise or fractal noise.

    Parameters:
    shape (tuple): Shape of the output array (h, w, c) or (h, w).
    min_inc (float): Minimum value for each octave.
    max_exc (float): Maximum value for each octave.
    octaves (tuple): Amplitude of each octave. The first octave has the highest resolution,
                     subsequent octaves have half the resolution of the previous octave.
    dtype (data-type): Desired data-type for the output array.

    Returns:
    np.array: Array of shape `shape` containing the fractal noise.
    """
    noises = []
    h, w = shape[:2]

    for i, octave in enumerate(octaves):
        noise_shape = (h // (1 << i), w // (1 << i)) + shape[2:]
        noise = np.random.uniform(low=min_inc, high=max_exc, size=noise_shape) * octave
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        noises.append(noise)

    return np.sum(noises, axis=0).astype(dtype)
