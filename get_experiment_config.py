# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:22:25 2024

@author: HCattan
"""

import numpy as np

def get_experiment_config():
    """
    Returns the configuration for the experiment.
    Users should modify the values in this function to suit their specific experiment setup.
    """
    # folder with the data
    #data_folder = r'D:\Social decision making task\'
    data_folder=r'D:\Social decision making task'
    # mice_id
    mice = ['M113B', 'M114A', 'M114C', 'M115A', 'M115D', 'M118A']
    empathic_port = ['left', 'right', 'right', 'right', 'left', 'left']
    altruistic_mice = [0, 1, 0, 1, 1, 1]
    days = [0, 1, 2, 3, 4]
    time_window = np.linspace(-2, 2, 20) * 1000  #in ms
    
    # Experiment settings
    equal_trials = 1   # get equal number of trials for empathic and selfish pokes per mouse
    recipient_features = 0  # include recipient SLEAP tracking
    previous_choice = 0  # include previous actor choice
    
    return data_folder, mice, empathic_port, altruistic_mice, days, time_window, equal_trials, recipient_features, previous_choice

# Call the configuration function
