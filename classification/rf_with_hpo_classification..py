import glob
import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from tqdm import tqdm

def split_data(csv_file_path, output_save):
    """
    Splits the data from a CSV file into training and testing datasets and saves them as separate CSV files.

    Parameters:
        csv_file_path (str): Path to the input CSV file.
        output_save (str): Directory where the train and test CSV files will be saved.

    Returns:
        tuple: x_train, x_test, y_train, y_test datasets.
    """
    csv_file = pd.read_csv(csv_file_path)

    # Drop the 'Image' column if it exists
    if 'Image' in csv_file.columns:
        csv_file.drop(columns=['Image'], inplace=True)

    x_source = csv_file.drop('label', axis='columns')
    y_source = csv_file['label']

    x_train, x_test, y_train, y_test = train_test_split(x_source, y_source, test_size=0.2, random_state=123)

    # Create DataFrames for train and test datasets
    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)

    # Save train and test datasets
    df_train.to_csv(os.path.join(output_save, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(output_save, 'test.csv'), index=False)

    return x_train, x_test, y_train, y_test

def get_random_grid(n_estimators_val, max_features, max_depth_val, min_samples_split_val, min_samples_leaf_val):
    """
    Generates a random grid of hyperparameters for RandomizedSearchCV.

    Parameters:
        n_estimators_val (list): List containing start, stop, and number of values for 'n_estimators'.
        max_features (list): List of options for 'max_features'.
        max_depth_val (list): List containing start, stop, and number of values for 'max_depth'.
        min_samples_split_val (list): List containing start, stop, and number of values for 'min_samples_split'.
        min_samples_leaf_val (list): List containing start, stop, and number of values for 'min_samples_leaf'.

    Returns:
        dict: Random grid of hyperparameters.
    """
    n_estimators = [int(x) for x in np.linspace(n_estimators_val[0], n_estimators_val[1], n_estimators_val[2])]
    max_depth = [int(x) for x in np.linspace(max_depth_val[0], max_depth_val[1], max_depth_val[2])]
    min_samples_split = [int(x) for x in np.linspace(min_samples_split_val[0], min_samples_split_val[1], min_samples_split_val[2])]
    min_samples_leaf = [int(x) for x in np.linspace(min_samples_leaf_val[0], min_samples_leaf_val[1], min_samples_leaf_val[2])]

    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'criterion': ['entropy', 'gini']
    }

    return random_grid

def get_best_model(n_iter, cv, verbose, random_state, n_jobs, random_grid, x_train, y_train, output_save):
    """
    Finds the best Random Forest model using RandomizedSearchCV and saves it as a pickle file.

    Parameters:
        n_iter (int): Number of parameter settings sampled.
        cv (int): Number of cross-validation folds.
        verbose (int): Controls verbosity of output.
        random_state (int): Random seed for reproducibility.
        n_jobs (int): Number of jobs to run in parallel.
        random_grid (dict): Random grid of hyperparameters.
        x_train (DataFrame): Training feature data.
        y_train (Series): Training target data.
        output_save (str): Directory to save the best model file.

    Returns:
        RandomForestClassifier: The best model found by RandomizedSearchCV.
    """
    rf = RandomForestClassifier()
    rf_randomcv = RandomizedSearchCV(
        estimator=rf,
        param_distributions=random_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs
    )

    rf_randomcv.fit(x_train, y_train)

    best_model = rf_randomcv.best_estimator_

    filename = f'{output_save}/RF_best'
    pickle.dump(best_model, open(filename, 'wb'))

    return best_model

def get_final_data(best_model, x_train, y_train, x_test, y_test, y_predict, output_save):
    """
    Analyzes and saves model performance, feature importance, and accuracy metrics.

    Parameters:
        best_model (RandomForestClassifier): The trained Random Forest model.
        x_train (DataFrame): Training feature data.
        y_train (Series): Training target data.
        x_test (DataFrame): Testing feature data.
        y_test (Series): Testing target data.
        y_predict (ndarray): Predicted target values.
        output_save (str): Directory to save analysis results.

    Returns:
        None
    """
    best_params_df = pd.DataFrame([best_model.get_params()])
    cm = confusion_matrix(y_test, y_predict)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    performance_metrics = {
        'Metric': ['Accuracy', 'Precision', 'Specificity', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, specificity, recall, f1]
    }
    df_performance = pd.DataFrame(performance_metrics)

    feature_importances = best_model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': x_test.columns.tolist(),
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    train_accuracy = best_model.score(x_train, y_train)
    test_accuracy = best_model.score(x_test, y_test)
    train_test_accuracy = {
        'Dataset': ['Train Accuracy', 'Test Accuracy'],
        'Accuracy': [train_accuracy, test_accuracy]
    }
    df_train_test_accuracy = pd.DataFrame(train_test_accuracy)
    df_train_test_accuracy.to_csv(os.path.join(output_save, 'test_train_accuracy.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=axes[0],
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')
    axes[0].set_title('Confusion Matrix')

    ax = sns.barplot(x='Metric', y='Value', data=df_performance, palette='Blues_d', ax=axes[1])
    axes[1].set_title('Performance Metrics')
    axes[1].set_ylabel('Value')
    axes[1].set_ylim(0, 1)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(f'{output_save}/confusion_matrix_and_performance_metrics.png', dpi=300)
    plt.close()

    best_params_df.to_csv(os.path.join(output_save, 'model_best_params.csv'), index=False)
    features_df.to_csv(os.path.join(output_save, 'features_ranking.csv'), index=False)

def otomatis(csv_file, output_path):
    """
    Automates the entire process of training and evaluating a Random Forest model.

    Parameters:
        csv_file (str): Path to the input CSV file.
        output_path (str): Directory to save all output files.

    Returns:
        None
    """
    os.makedirs(output_path, exist_ok=True)

    x_train, x_test, y_train, y_test = split_data(csv_file, output_path)

    random_grid = get_random_grid(
        n_estimators_val=[200, 10000, 100], 
        max_features=['sqrt', 'log2'],
        max_depth_val=[10, 100, 10], 
        min_samples_split_val=[2, 5, 10, 14], 
        min_samples_leaf_val=[1, 2, 4, 6, 8]
    )

    best_model = get_best_model(
        n_iter=100,
        cv=3,
        verbose=3,
        random_state=123,
        random_grid=random_grid,
        n_jobs=3,
        x_train=x_train,
        y_train=y_train,
        output_save=output_path,
    )

    y_predict = best_model.predict(x_test)

    get_final_data(best_model, x_train, y_train, x_test, y_test, y_predict, output_path)

def main():
    """
    Main function to process all CSV files in a folder and apply the automated workflow.

    Parameters:
        None

    Returns:
        None
    """
    folder_path = "../result_track/HASIL PERCOBAAN/S1K2P3REVISI"
    data_path = glob.glob(os.path.join(folder_path, '*.csv'))

    for data in tqdm(data_path, desc="Processing files", unit="file"):
        output_path = os.path.join(folder_path, os.path.basename(data)[:-4])
        otomatis(csv_file=data, output_path=output_path)

main()
