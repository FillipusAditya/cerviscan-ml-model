import pickle
import glob
import os
import warnings

import numpy as np
import xgboost as xgb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

def split_data(csv_file_path, output_save):
    """
    Splits the dataset into training and testing sets and saves them as CSV files.

    Args:
        csv_file_path (str): Path to the input CSV file.
        output_save (str): Directory where the split files will be saved.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    csv_file = pd.read_csv(csv_file_path)

    # Ensure the CSV file has a 'label' column
    if 'label' not in csv_file.columns:
        raise ValueError(f"Column 'label' not found in file {csv_file_path}")

    # Drop the 'Image' column if it exists
    if 'Image' in csv_file.columns:
        csv_file.drop(columns=['Image'], inplace=True)

    # Separate features (X) and target (Y)
    x_source = csv_file.drop('label', axis='columns')
    y_source = csv_file['label']

    # Split data into training and testing sets (80:20 ratio)
    x_train, x_test, y_train, y_test = train_test_split(x_source, y_source, test_size=0.2, random_state=123)

    # Map labels: 'abnormal' -> 1, 'normal' -> 0
    y_train = y_train.replace({'abnormal': 1, 'normal': 0})
    y_test = y_test.replace({'abnormal': 1, 'normal': 0})

    # Combine and save training and testing data
    df_train = pd.concat([x_train, y_train], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)

    df_train.to_csv(os.path.join(output_save, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(output_save, 'test.csv'), index=False)

    return x_train, x_test, y_train, y_test

def get_random_grid():
    """
    Generates a random grid for hyperparameter tuning.

    Returns:
        dict: A dictionary containing hyperparameter options for tuning.
    """
    random_grid = {
        'learning_rate': np.arange(0.01, 0.2, 0.01),
        'min_child_weight': np.arange(0, 5, 1),
        'min_split_loss': np.arange(0, 5, 1),
        'max_depth': np.arange(3, 10, 1),
        'reg_lambda': [2]
    }

    return random_grid

def get_best_model(cv, verbose, n_jobs, random_grid, x_train, y_train, output_save):
    """
    Performs grid search to find the best XGBoost model and saves it.

    Args:
        cv (int): Number of cross-validation folds.
        verbose (int): Verbosity level for grid search.
        n_jobs (int): Number of parallel jobs to run.
        random_grid (dict): Hyperparameter grid for tuning.
        x_train (DataFrame): Training features.
        y_train (Series): Training labels.
        output_save (str): Directory where the best model will be saved.

    Returns:
        XGBClassifier: The best XGBoost model.
    """
    xgb_model = xgb.XGBClassifier()
    xgb_grid_search = GridSearchCV(
        xgb_model,
        random_grid,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        scoring='accuracy')

    xgb_grid_search.fit(x_train, y_train)

    best_model = xgb_grid_search.best_estimator_

    filename = f'{output_save}/xgb_best'
    pickle.dump(best_model, open(filename, 'wb'))

    return best_model

def get_final_data(best_model, x_train, y_train, x_test, y_test, y_predict, output_save):
    """
    Generates final data including performance metrics, feature importance, and saves visualizations.

    Args:
        best_model (XGBClassifier): The trained model.
        x_train (DataFrame): Training features.
        y_train (Series): Training labels.
        x_test (DataFrame): Testing features.
        y_test (Series): Testing labels.
        y_predict (array): Predicted labels for the test set.
        output_save (str): Directory where results and visualizations will be saved.

    Returns:
        None
    """
    # Save best parameters
    best_params_df = pd.DataFrame([best_model.get_params()])

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_predict)

    # Calculate performance metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Create DataFrame for performance metrics
    performance_metrics = {
        'Metric': ['Accuracy', 'Precision', 'Specificity', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, specificity, recall, f1]
    }
    df_performance = pd.DataFrame(performance_metrics)

    # Calculate feature importances
    feature_importances = best_model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': x_test.columns.tolist(),
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Calculate train vs test accuracy
    train_accuracy = best_model.score(x_train, y_train)
    test_accuracy = best_model.score(x_test, y_test)
    train_test_accuracy = {
        'Dataset': ['Train Accuracy', 'Test Accuracy'],
        'Accuracy': [train_accuracy, test_accuracy]
    }
    df_train_test_accuracy = pd.DataFrame(train_test_accuracy)

    # Save train vs test accuracy
    df_train_test_accuracy.to_csv(os.path.join(output_save, 'test_train_accuracy.csv'), index=False)

    # Create visualizations
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

    # Save CSV files
    best_params_df.to_csv(os.path.join(output_save, 'model_best_params.csv'), index=False)
    features_df.to_csv(os.path.join(output_save, 'features_ranking.csv'), index=False)
    
folder_path = "FOLDER PATH"
data_path = glob.glob(os.path.join(folder_path, "file.csv"))
os.makedirs = os.path.join(folder_path, os.path.basename(data_path)[:-4])

