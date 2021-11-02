#!/usr/bin/env python3
import cv2
import numpy as np
from math import pi, cos, sin
import sys


def detect_edges(image):
    """Find edge points in a grayscale image.
    Args:
    - image (2D uint8 array): A grayscale image.
    Return:
    - edge_image (2D float array): A heat map where the intensity at each point
        is proportional to the edge magnitude.
    """
    row, column = image.shape
    edge_image = np.zeros((row, column))
    sober_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sober_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for i in range(row - 2):
        for j in range(column - 2):
            x = abs(np.sum(image[i:i + 3, j:j + 3] * sober_x))
            y = abs(np.sum(image[i:i + 3, j:j + 3] * sober_y))
            edge_image[i + 1, j + 1] = (x ** 2 + y ** 2) ** 0.5
    return edge_image


def hough_circles(edge_image, edge_thresh, radius_values):
    """Threshold edge image and calculate the hough transform accumulator array.

    Args:
    - edge_image (2D float array): An row x column heat map where the intensity at each
        point is proportional to the edge magnitude.
    - edge_thresh (float): A threshold on the edge magnitude values.
    - radius_values (1D int array): An array of R possible radius values.

    Return:
    - thresh_edge_image (2D bool array): Thresholded edge image indicating
        whether each pixel is an edge point or not.
    - accum_array (3D int array): rowough transform accumulator array. Should have
        shape R x row x column.
    """
    thresh_edge_image = edge_image >= edge_thresh
    row, column = thresh_edge_image.shape
    accum_array = np.zeros((len(radius_values), row, column), dtype=int)
    theta = np.arange(0, 2 * pi, 2 * pi / 360)
    dx_dy = []
    for i, r in enumerate(radius_values):
        for t in theta:
            dx_dy.append((i, int(r * cos(t)), int(r * sin(t))))
    for i in range(row):
        for j in range(column):
            if thresh_edge_image[i, j] == 0:
                continue
            else:
                for (r_ind, dx, dy) in dx_dy:
                    x = j + dx
                    y = i + dy
                    if 0 <= x < column and 0 <= y < row:
                        accum_array[r_ind, y, x] += 1
    # time2 = time()
    # print(time2-time1, '\n')
    # print(np.unique(accum_array))
    return thresh_edge_image, accum_array


def find_circles(image, accum_array, radius_values, hough_thresh):
    """Find circles in an image using output from rowough transform.
    Args:
    - image (3D uint8 array): An row x column x 3 BGR color image. rowere we use the
        original color image instead of its grayscale version so the circles
        can be drawn in color.
    - accum_array (3D int array): rowough transform accumulator array having shape
        R x row x column.
    - radius_values (1D int array): An array of R radius values.
    - hough_thresh (int): A threshold of votes in the accumulator array.
    Return:
    - circles (list of 3-tuples): A list of circle parameters. Each element
        (r, y, x) represents the radius and the center coordinates of a circle
        found by the program.
    - circle_image (3D uint8 array): A copy of the original image with detected
        circles drawn in color.
    """
    circles = []
    circle_image = image.copy()
    r_r, row, column = accum_array.shape
    for r_ind in range(r_r):
        for y in range(row):
            for x in range(column):
                if accum_array[r_ind, y, x] > hough_thresh:
                    if accum_array[r_ind, y, x] == \
                            np.max(accum_array[max(r_ind - 6, 0):min(r_ind + 7, r_r), max(y - 6, 0):min(y + 7, row),
                                   max(x - 6, 0):min(x + 7, column)]):
                        circles.append((radius_values[r_ind], y, x))
                        cv2.circle(circle_image, (x, y), radius_values[r_ind], (0, 255, 0), 2)
    return circles, circle_image


def main(argv):
    img_name = argv[0]
    thresh_val1 = int(argv[1])
    thresh_val2 = float(argv[2])
    radius_lb = int(argv[3])
    radius_ub = int(argv[4])
    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_edges = detect_edges(gray_image)
    e_thresh, acc = hough_circles(image_edges, thresh_val1, np.arange(radius_lb, radius_ub))
    circles, circle_image = find_circles(img, acc, np.arange(radius_lb, radius_ub), np.amax(acc) * thresh_val2)
    print(circles)
    cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
    cv2.imwrite('output/' + img_name + "_edges.png", e_thresh * 255)
    cv2.imwrite('output/' + img_name + '_circles.png', circle_image)


if __name__ == '__main__':
    # python p2_hough_circles.py "coins" 255 0.5 20 40
    # python p2_hough_circles.py "humanerythrocyteslarge" 255 0.5 18 25
    main(sys.argv[1:])

