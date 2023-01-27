"""
Functions to generate simple ensembles of shapes.
These functions return lists of binary masks.
One mask per member of the ensemble.
"""

import numpy as np
from skimage.draw import disk, rectangle, line, polygon
from skimage.transform import rotate

def get_circles_dataset(num_cols, num_rows, num_members, variation="location"):
    center_coords = np.zeros((num_members, 2)) + (num_cols / 2)
    radii = np.repeat(50, num_members)
    if variation == "location":
        center_coords = (np.random.randn(num_members, 2) * 20) + num_cols / 2
    if variation == "radii":
        radii = np.random.randn(num_members) * 50

    circle_rr_cc = [disk(center_coords[i], radii[i], shape=(num_rows, num_cols)) for i in range(num_members)]

    circle_masks = [np.zeros((num_rows, num_cols)) for _ in range(num_members)]
    for i, (rr, cc) in enumerate(circle_rr_cc):
        circle_masks[i][rr, cc] = 1

    return circle_masks


def get_rectangles_dataset(num_cols, num_rows, num_members, variation="location"):
    center_coords = np.zeros((num_members, 2)) + (num_cols / 2)
    size_a = np.repeat(100, num_members)
    size_b = np.repeat(100, num_members)
    rotation = np.repeat(0, num_members)
    if variation == "location":
        center_coords = (np.random.randn(num_members, 2) * 50) + num_cols / 2
    if variation == "size":
        size_a = np.random.randn(num_members) * 100
        size_b = np.random.randn(num_members) * 100
    if variation == "rotation":
        rotation = np.random.randint(-20, 20, num_members)

    size_a = size_a.reshape((-1, 1))
    size_b = size_b.reshape((-1, 1))
    rotation = rotation.reshape((-1, 1))

    starts = center_coords - (np.concatenate([size_a, size_b], axis=1)/2)
    ends = center_coords + (np.concatenate([size_a, size_b], axis=1) / 2)

    rectangle_rr_cc = [[l.astype(np.int) for l in rectangle(starts[i], ends[i], shape=(num_rows, num_cols))] for i in range(num_members)]

    rectangle_masks = [np.zeros((num_rows, num_cols)) for _ in range(num_members)]
    for i, (rr, cc) in enumerate(rectangle_rr_cc):
        rectangle_masks[i][rr, cc] = 1
        rectangle_masks[i] = rotate(rectangle_masks[i], rotation[i][0])

    return rectangle_masks


def get_circles_with_shape_outliers(num_cols, num_rows, num_members, fraction_outliers=0.1):
    center_coords = np.random.randn(num_members, 2) * 10 + (num_cols / 2)
    radii = np.random.randint(num_cols//4, num_cols//4 + 10, num_members)
    circle_rr_cc = [disk(center_coords[i], radii[i], shape=(num_rows, num_cols)) for i in range(num_members)]

    small_circle_rr_cc = []
    for i in range(num_members):
        if np.random.random(1)[0] < fraction_outliers:
            angle = 45
            cc_y = center_coords[i, 0] + radii[i] * np.sin(np.deg2rad(angle))
            cc_x = center_coords[i, 1] + radii[i] * np.cos(np.deg2rad(angle))
            cr = radii[i] * 0.3

            small_circle_rr_cc.append(disk((cc_y, cc_x), cr, shape=(num_rows, num_cols)))
        else:
            small_circle_rr_cc.append(None)

    circle_masks = [np.zeros((num_rows, num_cols)) for _ in range(num_members)]
    for i, (rr, cc) in enumerate(circle_rr_cc):
        circle_masks[i][rr, cc] = 1
        if small_circle_rr_cc[i] is not None:
            circle_masks[i][small_circle_rr_cc[i][0], small_circle_rr_cc[i][1]] = 1

    return circle_masks


def get_shapes_dataset(num_cols, num_rows, num_members, fraction_squares=1/3, fraction_circles=1/3, fraction_triangles=1/3):
    shape_type = np.zeros(num_members)
    shapes_rr_cc = []
    center = np.array((num_cols // 2, num_rows //2))
    for st in shape_type:
        random_number = np.random.random()
        if 0 <= random_number < fraction_squares:
            start = center - np.random.randint(90, 110, 1)[0]
            end = center + np.random.randint(90, 110, 1)[0]
            shapes_rr_cc.append(rectangle(start, end, shape=(num_rows, num_cols)))
        elif fraction_squares <= random_number < fraction_squares + fraction_circles:
            radius = np.random.randint(90, 110, 1)[0]
            shapes_rr_cc.append(disk((center[0], center[1]), radius, shape=(num_rows, num_cols)))
        elif fraction_squares + fraction_circles <= random_number < fraction_squares + fraction_circles + fraction_triangles:
            poly_coords = np.array((
                (50, center[0]),
                (250, num_cols // 3),
                (250, (num_cols // 3) * 2)
            ))
            shapes_rr_cc.append(polygon(poly_coords[:,0], poly_coords[:,1], shape=(num_rows, num_cols)))

    shape_masks = [np.zeros((num_rows, num_cols)) for _ in range(num_members)]
    for i, (rr, cc) in enumerate(shapes_rr_cc):
        shape_masks[i][rr, cc] = 1

    return shape_masks

def get_lines_dataset(num_cols, num_rows, num_members, variation="radial"):
    center = (0, num_cols//2)
    angles_deg = np.linspace(0, 50, num_members)

    lines_rr_cc = []
    for i, adeg in enumerate(angles_deg):
        if variation == "radial":
            r0 = int(center[0])
            r1 = int(center[1])
            r1 = int(num_cols//3 * np.cos(np.deg2rad(adeg)) + center[0])
            c1 = int(num_cols//3 * np.sin(np.deg2rad(adeg)) + center[1])
        if variation == "equispaced":
            r0 = r1 = int(10 + i * ((num_rows - 20) // num_members))
            c0 = 10
            c1 = int(num_cols - 10)
        lines_rr_cc.append([a.astype(np.int) for a in line(r0, c0, r1, c1)])

    lines_masks = [np.zeros((num_rows, num_cols)) for _ in range(num_members)]
    for i, (rr, cc) in enumerate(lines_rr_cc):
        lines_masks[i][rr, cc] = 1

    return lines_masks