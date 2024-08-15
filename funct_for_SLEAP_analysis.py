# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:43:52 2024

@author: HCattan
"""
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d

def gaussian_smoothing(data, sigma=3, axis=0):
    """
    Applies Gaussian smoothing to a 1D array or a 2D matrix.
    
    Parameters:
    data (numpy array): The input data array or matrix to be smoothed.
    sigma (float): The standard deviation of the Gaussian kernel. Default is 3.
    axis (int): The axis along which to apply the smoothing (0 for rows, 1 for columns). Default is 0.
    
    Returns:
    numpy array: The smoothed data.
    """
    if data.ndim == 1:
        # If data is a 1D array
        smoothed_data = gaussian_filter1d(data, sigma)
    elif data.ndim == 2:
        # If data is a 2D matrix, apply smoothing along the specified axis
        smoothed_data = np.apply_along_axis(gaussian_filter1d, axis, data, sigma)
    else:
        raise ValueError("Input data must be a 1D array or a 2D matrix.")
    
    return smoothed_data


def find_indices(node_names, names_to_find):
    """
    Find indices of specific names in a list of node names.

    Parameters:
    - node_names: List of node names.
    - names_to_find: List of names to find in node_names.

    Returns:
    - Dictionary with names as keys and their indices in node_names as values.
    """
    indices = {}
    for name in names_to_find:
        try:
            indices[name] = node_names.index(name)
        except ValueError:
            indices[name] = None  # Indicate that the item was not found
    return indices

def sort_locations_by_node_order(locations_dict, node_names_dict):
    """
    Sort locations in the dictionary by the node order.

    Parameters:
    - locations_dict: Dictionary with indices as keys and location arrays as values.
    - node_names_dict: Dictionary with indices as keys and node name lists as values.

    Returns:
    - Tuple of (sorted node names dictionary, sorted locations dictionary).
    """
    sorted_locations_dict = {}
    for idx in locations_dict:
        node_order = node_names_dict[idx]
        node_index_map = {node_name: i for i, node_name in enumerate(node_order)}
        locations_dict[idx] = locations_dict[idx][:, [node_index_map[node_name] for node_name in node_order], :]
        node_names_dict[idx] = sorted(node_names_dict[idx])
    return node_names_dict, locations_dict

def calculate_rectangle_center(x0, y0, length, width):
    """
    Calculate the center of a rectangle.

    Parameters:
    - x0, y0: Coordinates of the bottom-left corner of the rectangle.
    - length: Length of the rectangle.
    - width: Width of the rectangle.

    Returns:
    - Tuple of (x, y) coordinates of the center.
    """
    x_c = x0 + length / 2
    y_c = y0 + width / 2
    return x_c, y_c

def calculate_angle_between_vectors(v1, v2):
    """
    Calculate the angle between two vectors.

    Parameters:
    - v1, v2: Input vectors.

    Returns:
    - Angle between the vectors in radians.
    """
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip to avoid numerical issues
    return angle

def angle_between_point_and_line(line_point1, line_point2, mouse_angles):
    """
    Calculate the angle between the mouse's head direction and the perpendicular to the line.
    
    Parameters:
    - line_point1: numpy array of shape (2,), first extremity point of the line (x1, y1).
    - line_point2: numpy array of shape (2,), second extremity point of the line (x2, y2).
    - mouse_angles: numpy array of shape (n,), angles of the mouse's head in radians.
    
    Returns:
    - angles_in_degrees: numpy array of shape (n,), angles between the mouse's head direction and the perpendicular to the line in degrees.
    """
    # Calculate the direction vector of the line
    direction_vector = line_point2 - line_point1
    
    # Calculate the perpendicular vector to the line
    perpendicular_vector = np.array([-(direction_vector[1]), direction_vector[0]])
    
    # Normalize the perpendicular vector
    perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
    
    angles = []
    
    for angle in mouse_angles:
        # Calculate the mouse's direction vector
        mouse_direction_vector = np.array([np.cos(angle), np.sin(angle)])
        
        # Calculate the angle between the perpendicular vector and the mouse's direction vector
        cos_theta = np.dot(perpendicular_vector, mouse_direction_vector)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Ensure the value is within [-1, 1] to avoid numerical issues
        angles.append(theta)
    
    # Convert the angles to degrees
    angles_in_degrees = np.degrees(angles)
    
    return np.array(angles_in_degrees)


def head_angles(nose, left_ear, right_ear):
    """
    Calculate the head angles based on nose and ear positions.

    Parameters:
    - nose, left_ear, right_ear: numpy arrays of shape (n, 2), positions of the nose, left ear, and right ear respectively.

    Returns:
    - Tuple of (cos_angle, sin_angle, angle in degrees).
    """
    head_direction = nose - (left_ear + right_ear) / 2
    cos_angle = head_direction[:, 0] / np.linalg.norm(head_direction, axis=1)
    sin_angle = head_direction[:, 1] / np.linalg.norm(head_direction, axis=1)
    angle = np.degrees(np.arctan2(head_direction[:, 0], head_direction[:, 1]))
    return cos_angle, sin_angle, angle

def velocity(locations, delta_t):
    """
    Calculate the velocity based on location data.

    Parameters:
    - locations: numpy array of shape (n, 2), positions.
    - delta_t: Time difference between frames.

    Returns:
    - numpy array of shape (n-1, 2), velocities.
    """
    vel = np.diff(locations, axis=0) / delta_t
    return vel

def crop_data_around_indices(data, LED_ON, cut_indices, timeStamps):
    """
    Crop data around indices of interest.

    Parameters:
    - data: Input data array.
    - LED_ON: Dictionary with labels and corresponding indices.
    - cut_indices: List of start and end indices for cropping.

    Returns:
    - Tuple of (cropped data array, labels list).
    """
    cropped_data = []
    labels = []

    for label, indices in LED_ON.items():
        indices = np.squeeze(np.asarray(indices, dtype=int))

        for idx in indices:
            # find 
            find_first_trueindex = [
            (timeStamps['Time Stamp (ms)'] > round(timeStamps['Time Stamp (ms)'][idx] + cut_indices[x])).idxmax()
            for x in range(2)
            ]          
            
            start_idx = find_first_trueindex[0]
            end_idx = find_first_trueindex[1]
            cropped_segment = np.mean(data[start_idx:end_idx])
            cropped_data.append(cropped_segment)
            labels.append(label)

    return np.array(cropped_data), labels

def interpolate_crop(data, LED_on, cut_indices, timeStamps):
    """
    Interpolate data and crop it around specified indices.

    Parameters:
    - data: List of data arrays.
    - LED_on: Dictionary with labels and corresponding indices.
    - cut_indices: List of start and end indices for cropping.

    Returns:
    - Tuple of (interpolated and cropped data list, labels list).
    """
    data_trials = []
    for i in range(len(data)):
        #data[i] = pd.Series(data[i]).interpolate(method='linear')
        data_trials_1, labels = crop_data_around_indices(data[i], LED_on, cut_indices,timeStamps)
        data_trials.append(data_trials_1)
    
    return data_trials, labels
