# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:17:23 2024

@author: HCattan
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.io  # to load MATLAB files
from sklearn.model_selection import train_test_split, cross_val_score , cross_val_predict
from sklearn import svm
from funct_for_SLEAP_analysis import (
    sort_locations_by_node_order,
    head_angles,
    calculate_rectangle_center,
    calculate_angle_between_vectors,
    angle_between_point_and_line,
    velocity,
    crop_data_around_indices,
    interpolate_crop,
    find_indices,
    gaussian_smoothing
)

def prepare_behavioral_data(data_folder, mouse, day, empathic_port, window, equal_trials, recipient_features,previous_choice):
    """
    Prepare behavioral data for analysis.
    
    Parameters:
    - data_folder: str, path to the data folder.
    - mouse: str, identifier for the mouse.
    - day: int, day of the experiment.
    - empathic_port: int, indicator for the empathic port (0 or 1) (1= right, 0=left)
    - window: int, window size for data processing.
    - equal_trials: bool, whether to balance trials between conditions.
    - recipient_features: bool, whether to include feature of actor x recipient, or only actor
    
    Returns:
    - df: pandas DataFrame, prepared data.
    """
    # Upload h5 files (SLEAP) for actor and recipient
    #file_names = [f'{data_folder}/{mouse}_day{day}.analysis.h5', f'{data_folder}/{mouse}_day{day}_recip.analysis.h5']
    file_name_actor=f'{data_folder}\{mouse}_day{day}.analysis.h5'
    
    # Define dictionaries to store data from each file
    locations_dict = {}
    node_names_dict = {}

    with h5py.File(file_name_actor, "r") as f:
        locations = f["tracks"][:].T  # Assuming "tracks" is the dataset you want to extract, transposing for desired shape
        node_names = [n.decode() for n in f["node_names"][:]]  # Decoding node names assuming they are stored as bytes
        # Store data in dictionaries with keys based on file index
        locations_dict = np.squeeze(locations)

    print(len(locations_dict))  # Number of instances (in our case mice)
    print(locations_dict.shape)  # Size of the locations for each instance
    names_to_find=['l_ear', 'r_ear', 'nose', 'neck', 'tail']
    indices = find_indices(node_names, names_to_find)
    
    for name, index in indices.items():
        if index is not None:
            print(f"Index of '{name}': {index}")
        else:
            print(f"'{name}' not found in the list.")


    for k in range(len(names_to_find)):
        locations_dict[:, k, :] = pd.DataFrame(locations_dict[:, k, :]).interpolate(method='linear')

    sigma=6
    R_ear_loc =  gaussian_smoothing(locations_dict[:, indices['r_ear'], :], sigma)
    L_ear_loc = gaussian_smoothing(locations_dict[:, indices['l_ear'], :], sigma)
    nose_loc = gaussian_smoothing(locations_dict[:, indices['nose'], :], sigma)
    body_loc = gaussian_smoothing(locations_dict[:, indices['neck'], :], sigma)


    # Import the time (frames) of the trials with pokes
    filename = f'{data_folder}/{mouse}_day{day}_position.mat'
    mat = scipy.io.loadmat(filename)
    filename_ts= f'{data_folder}/{mouse}_day{day}_timeStamps.csv'
    timeStamps=pd.read_csv(filename_ts)
    print(timeStamps.columns)
    timeStamps=timeStamps.drop(columns=['Buffer Index'])

    # Extract LED_on frames
    LED_on_R = mat['LED_on_frames_R']
    LED_on_L = mat['LED_on_frames_L']

    if empathic_port=='right':
        LED_ON = {
            'empathic': LED_on_R,
            'selfish': LED_on_L
        }
    elif empathic_port=='left':
         LED_ON = {
            'empathic': LED_on_L,
            'selfish': LED_on_R
        }

    food_port = mat['foodport_pos']
    food_p_r = np.mean(food_port[0, 0], axis=0)
    div_pos = mat['divider_pos']
    div_pos_start = np.array([div_pos[0, 0][0], div_pos[0, 1][0]])
    div_pos_stop = np.array([div_pos[1, 0][0], div_pos[0, 1][1]])

    food_p_act = calculate_rectangle_center(food_port[0, 1][0, 0], food_port[0, 1][1, 0], food_port[0, 1][0, 1], food_port[0, 1][1, 1])

    # Calculate features of the actor
    cos_angle, sin_angle, angle_nose = head_angles(nose_loc, L_ear_loc, R_ear_loc)
    angles = angle_between_point_and_line(div_pos_start, div_pos_stop, angle_nose)

    # Assuming delta_t is known (time difference between frames)
    delta_t = 1  # Example time difference between frames
    velocity_actor = np.squeeze(velocity(nose_loc, delta_t))

    dist_nose_to_food_act = np.linalg.norm(nose_loc - food_p_act, axis=1)
    dist_nose_to_food_recip = np.linalg.norm(nose_loc - food_p_r, axis=1)
    angle_nose_to_divider = angle_between_point_and_line(div_pos_start, div_pos_stop, angle_nose)
    dist_nose_to_divider= abs(nose_loc[:,1] - (div_pos_stop[1]))

    X = {}
    X[0] = dist_nose_to_food_act
    X[1] = dist_nose_to_food_recip
    X[2] = dist_nose_to_divider
    X[3] = np.cos(angle_nose_to_divider)
    X[4] = np.sin(angle_nose_to_divider)
    headers=["dist nose to foodport", "dist nose to recip foodport", "dist nose to divider", 
             "cos nose to divider", "sin nose to divider" ]

    if recipient_features==1:
        file_name_recip=[f'{data_folder}/{mouse}_day{day}_recip.analysis.h5']
        with h5py.File(file_name_recip, "r") as f:
            locations = f["tracks"][:].T  # Assuming "tracks" is the dataset you want to extract, transposing for desired shape
            node_names = [n.decode() for n in f["node_names"][:]]  # Decoding node names assuming they are stored as bytes
            locations_dict[1] = np.squeeze(locations)

        names_to_find=['l_ear', 'r_ear', 'nose', 'neck', 'tail']
        indices = find_indices(node_names, names_to_find)
        
        for name, index in indices.items():
            if index is not None:
                print(f"Index of '{name}': {index}")
            else:
                print(f"'{name}' not found in the list.")
            
        R_ear_loc_recip = locations_dict[:, indices['r_ear'], :]
        L_ear_loc_recip = locations_dict[:, indices['l_ear'], :]
        nose_loc_recip = locations_dict[:, indices['nose'], :]
        body_loc_recip = locations_dict[:, indices['neck'], :]

    
        
    # Calculate features of the actor towards the recipient
        direction_noses = nose_loc_recip - nose_loc
        cos_noses = direction_noses[:, 0] / np.linalg.norm(direction_noses, axis=1)
        sin_noses = direction_noses[:, 1] / np.linalg.norm(direction_noses, axis=1)
    
        # Calculate the distance between the two noses
        dist_nose_to_nose = np.linalg.norm(nose_loc - nose_loc_recip, axis=1)
        # Calculate the distance between the actor nose and the body
        dist_body_to_nose = np.linalg.norm(nose_loc - body_loc_recip, axis=1)
    
        X[6] = cos_noses
        X[7] = sin_noses
        X[8] = dist_nose_to_nose
        X[9] = dist_body_to_nose
        
        additional_headers=["cos between noses", "sin between noses", "dist nose to nose", "dist body to nose"]
        headers.extend(additional_headers)

    if previous_choice==1:
        # Concatenate the data
        data = np.concatenate((LED_ON["empathic"][0], LED_ON["selfish"][0]))
        # Create the labels
        labels_port = np.concatenate((np.ones(len(LED_ON["empathic"][0])), np.zeros(len(LED_ON["selfish"][0]))))
        # Combine data and labels into time_poking array
        time_poking = np.column_stack((data, labels_port))
        time_poking = time_poking[np.argsort(time_poking[:, 0])]
        previous_port=time_poking[1::,1]
        first_poking=time_poking[0,1]
    
    
    
    
    # Prepare variable including all the features of interest
    data_trials, labels = interpolate_crop(X, LED_ON, [window[0], window[1]], timeStamps)
    XX = np.squeeze(data_trials)
    labels = np.array(labels)
    
    if previous_choice==1:
        if first_poking==0:
           trial_to_remove=np.min( np.where(labels=="selfish"))
           labels=np.delete(labels,trial_to_remove)
           XX=np.delete(XX, trial_to_remove, axis=1)
        else:
            trial_to_remove=np.min( np.where(labels=="empathic"))
            labels=np.delete(labels,trial_to_remove)
            XX=np.delete(XX, trial_to_remove, axis=1) 
        XX[XX.shape[1],:]=previous_port
         
           
           
    port_label =[ empathic_port] * len(labels)
    X_extended = np.vstack([XX, labels, port_label]).T
    
    
    if equal_trials==1:
        empathic_indices = np.where(X_extended == 'empathic')[0]
        selfish_indices = np.where(X_extended == 'selfish')[0]

        min_samples = min(len(empathic_indices), len(selfish_indices))
        selected_indices = np.concatenate([empathic_indices[:min_samples], selfish_indices[:min_samples]])

        X_balanced = (X_extended[selected_indices,:])
        del X_extended
        X_extended=X_balanced
    
    headers.append("labels")
    headers.append("port_label")
    # Create DataFrame with headers
    df = pd.DataFrame(X_extended, columns=headers)
    

    return df
