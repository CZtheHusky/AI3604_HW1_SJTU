#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from math import sin, cos, pi
from collections import defaultdict


def binarize(gray_image, thresh_val):
    # TODO: 255 if intensity >= thresh_val else 0
    binary_image = np.zeros(gray_image.shape)
    binary_image[gray_image > thresh_val] = 255
    return np.uint8(binary_image)


def label(binary_image):
    # TODO
    label = np.zeros(binary_image.shape)
    labeled_image = np.zeros(binary_image.shape)
    counter = 0
    (row, column) = binary_image.shape
    match = defaultdict(int)
    for x in range(row):
        for y in range(column):
            if binary_image[x, y] != 0:
                flat = np.unique(label[max(0, x - 2):min(x + 3, row), max(0, y - 2):min(y + 3, column)])
                if flat[-1] == 0:
                    counter += 1
                    label[x][y] = counter
                elif flat[-2] != 0:
                    for i in flat:
                        if i != 0:
                            minimum = i
                            for j in flat:
                                if j != 0 and j != minimum:
                                    if match[j] == 0 or match[j] > minimum:
                                        match[j] = minimum
                            label[x][y] = minimum
                            break
                else:
                    label[x][y] = flat[-1]
    for i in sorted(match.keys(), reverse=True):
        label[label == i] = match[i]
    label_count = list(set(np.ravel(label)))
    label_count.sort()
    counter = len(label_count)
    for i in range(1, counter):
        labeled_image[label == label_count[i]] = int(i / (counter - 1) * 255)
    return np.uint8(labeled_image)


def get_attribute(labeled_image):
    # TODO
    labels, counts = np.unique(labeled_image, return_counts=True)
    labels = labels[1:]
    counts = counts[1:]
    attribute_list = []
    for i, label in enumerate(labels):
        S = counts[i]
        ys, xs = np.where(labeled_image == label)
        ys = (labeled_image.shape[0] - 1) - ys
        x_center = sum(xs) / S
        y_center = sum(ys) / S
        a, b, c = 0, 0, 0
        x_normed, y_normed = xs - x_center, ys - y_center
        for x, y in zip(x_normed, y_normed):
            a += x ** 2
            b += 2 * x * y
            c += y ** 2
        max_xs = x_normed.max()
        max_ys = y_normed.max()
        size = pow(max_xs ** 2 + max_ys ** 2, 0.5)
        theta_1 = np.arctan2(b, a - c) / 2
        theta_2 = theta_1 + pi / 2
        if theta_1 < 0:
            theta_1 += pi
        E_min = a * sin(theta_1) ** 2 - b * sin(theta_1) * cos(theta_1) + c * cos(theta_1) ** 2
        E_max = a * sin(theta_2) ** 2 - b * sin(theta_2) * cos(theta_2) + c * cos(theta_2) ** 2
        roundedness = E_min / E_max
        attribute_list.append(
            {'label': round(label, 4), 'position': {'x': round(float(x_center), 4), 'y': round(float(y_center), 4)},
             'orientation': round(float(theta_1), 4),
             'roundedness': round(float(roundedness), 4), 'size': round(size, 4)})
    return attribute_list


def main(argv):
    img_name = argv[0]
    thresh_val = int(argv[1])
    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_image = binarize(gray_image, thresh_val=thresh_val)
    labeled_image = label(binary_image)
    attribute_list = get_attribute(labeled_image)
    point_size = 2
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 0 、4、8
    for i in attribute_list:
        x = int(i['position']['x'])
        y = img.shape[0] - 1 - int(i['position']['y'])
        theta = i['orientation']
        size = i['size']
        ori_image = cv2.circle(img, (x, y), point_size, point_color, thickness)
        ori_image = cv2.line(ori_image, (x - int(size * cos(-theta)), y - int(size * sin(-theta))),
                             (x + int(size * cos(-theta)), y + int(size * sin(-theta))), color=(0, 255, 0),
                             thickness=2)
    cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
    cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
    cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
    cv2.imwrite('output/' + img_name + "_d.png", ori_image)
    print(img_name + ':')
    [print(i) for i in attribute_list]


if __name__ == '__main__':
    # python p1_object_attributes.py "coins" 128
    main(sys.argv[1:])

# argv = ['many_objects_1', 128]
# main(argv)
# argv = ['coins', 128]
# main(argv)
# argv = ['many_objects_2', 128]
# main(argv)
# argv = ['two_objects', 128]
# main(argv)
