import cv2
import numpy as np

def fast_thinning(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = img - open
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

def find_distance(x0, y0, direction, binary_img):
    step_size = 8
    current_dist = 0
    dir_x, dir_y = direction
    while step_size >= 1:
        new_x = int(x0 + dir_x * (current_dist + step_size))
        new_y = int(y0 + dir_y * (current_dist + step_size))
        if (0 <= new_x < binary_img.shape[1] and 
            0 <= new_y < binary_img.shape[0] and 
            binary_img[new_y, new_x]):
            current_dist += step_size
        else:
            step_size //= 2
    return current_dist

def calculate_angle(dir1, dir2):
    """Calcula el ángulo entre dos direcciones en grados"""
    dot_product = dir1[0]*dir2[0] + dir1[1]*dir2[1]
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle_rad)
