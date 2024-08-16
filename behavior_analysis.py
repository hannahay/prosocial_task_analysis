# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:49:11 2024

@author: HCattan
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# Import the custom modules
from prepare_behavioral_data_NEW import prepare_behavioral_data
from get_experiment_config import get_experiment_config

# Get settings from the user
data_folder, mice, empathic_port, altruistic_mice, days, time_window, equal_trials, recipient_features, previous_choice = get_experiment_config()

fig_path=r'D:\Social decision making task\figures'
if not os.path.exists(fig_path):
    # Create the directory if it does not exist
    os.makedirs(folder)
    print(f"Directory created at: {folder}")
    
    
# Initialize results data structures
results_dict = {}
results_dict_shuffled = {}

# Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM (RBF)': SVC(kernel='rbf', gamma='scale'),
    'SVM (Sigmoid)': SVC(kernel='linear', gamma='scale'),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(),
}

num_of_shuffles=50
results = pd.DataFrame(index=np.linspace(0, len(time_window)-1, len(time_window)), columns=classifiers.keys())
results_shuffled = pd.DataFrame(index=np.linspace(0, len(time_window)-1, len(time_window)), columns=classifiers.keys())


# Main loop over days and time windows
for day in range(len(days)):
    c = 0
    score_class = []
    score_class_shuffled = []

    for t in range(0, len(time_window) - 1):
        df = pd.DataFrame()

        for i in range(len(mice)):
            data_mouse = prepare_behavioral_data(data_folder, mice[i], day, empathic_port[i], 
                                                 time_window[[t, t + 1]], equal_trials, 
                                                 recipient_features, previous_choice)
            nan_values = ['NaN', 'nan', 'N/A', 'NULL', '']  # Add any other representations you need to handle
            data_mouse.replace(nan_values, np.nan, inplace=True)
            data_mouse.dropna(inplace=True)
            df = pd.concat([df, data_mouse])

        # Balance the dataset
        selfish_left = df[(df['labels'] == 'selfish') & (df['port_label'] == 'left')]
        selfish_right = df[(df['labels'] == 'selfish') & (df['port_label'] == 'right')]
        empathic_left = df[(df['labels'] == 'empathic') & (df['port_label'] == 'left')]
        empathic_right = df[(df['labels'] == 'empathic') & (df['port_label'] == 'right')]
        sample_size = min(len(selfish_left), len(selfish_right), len(empathic_left), len(empathic_right))
        random_df = pd.concat([selfish_left.sample(n=sample_size, replace=False),
                               selfish_right.sample(n=sample_size, replace=False),
                               empathic_left.sample(n=sample_size, replace=False),
                               empathic_right.sample(n=sample_size, replace=False)], axis=0)
        random_df.reset_index(drop=True, inplace=True)

        X = random_df.drop(["labels", "port_label"], axis=1)
        y = random_df["labels"]
        X_final, y_final = shuffle(X, y)  # Shuffle data and labels

        X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_train = pd.DataFrame(X_train, columns=X.columns)

        time_point_scores = {}
        # Run cross-validation for each classifier
        for name, clf in classifiers.items():
            scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
            time_point_scores[name] = np.mean(scores)
            results.at[c, name] = time_point_scores[name]

        # Shuffle scores
            scores_shuffle = []
            time_point_scores_shuffled = {}

            for m in range(num_of_shuffles):
                y_final_shuffled = y_final.sample(frac=1).reset_index(drop=True)
                X_train, X_test, y_train_shuffle, y_test_shuffle = train_test_split(X_final, y_final_shuffled, test_size=0.2)
                X_train = sc.fit_transform(X_train)
                X_train = pd.DataFrame(X_train, columns=X.columns)
                clf.fit(X_train, y_train_shuffle)
                X_test = sc.transform(X_test)
                y_pred = clf.predict(X_test)
                scores_shuffle.append(f1_score(y_test_shuffle, y_pred, average='binary', pos_label='empathic' ))
                
            time_point_scores_shuffled[name]=np.mean(scores_shuffle)

            results_shuffled.at[c, name]= time_point_scores_shuffled[name]
            
        c += 1

    key = f'day_{day}'  # Create the dynamic key
    results_dict[key] = results
    results_dict_shuffled[key] = results_shuffled

# Plot results
    for d in results.columns:
        plt.figure()
        plt.plot(time_window, results[d])
        plt.plot(time_window,results_shuffled[d])
        plt.title(f'{d} Day {day}')
        plt.xlabel('Time (s)')
        plt.ylabel('Accuracy')
        name_fig= os.path.join(fig_path, f'{d} day {day}.png')
        plt.savefig(name_fig, format='png', dpi=300)
        plt.show()
