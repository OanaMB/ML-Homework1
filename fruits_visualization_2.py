import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from fruits_preprocessing_2 import getAllFruits
from collections import Counter

def plot_class_distribution_seaborn(y_train, y_test, fruits):
    # Convert numerical labels back to fruit names
    fruit_names_train = [fruits[label] for label in y_train]
    fruit_names_test = [fruits[label] for label in y_test]

    # Create count plots for training and test sets
    plt.figure(figsize=(42, 10))
    
    # Training set plot
    plt.subplot(1, 1, 1)
    sns.countplot(x=fruit_names_train, order=fruits)
    plt.xticks(rotation=90, fontsize=8)
    plt.title('Training Set Class Distribution')
    plt.xlabel('Fruit Class')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('./fruits/training_class_distribution.png', dpi=500)

    plt.figure(figsize=(42, 10))

    # Test set plot
    plt.subplot(1, 1, 1)
    sns.countplot(x=fruit_names_test, order=fruits)
    plt.xticks(rotation=90, fontsize=8)
    plt.title('Test Set Class Distribution')
    plt.xlabel('Fruit Class')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('./fruits/test_class_distribution.png', dpi=500)
 
    

def main():
    # Load transformed data
    transformed_data = np.load('./fruits/transformed_data_fruits_2.npz')
    X_train_pca = transformed_data['X_train_selected']
    X_test_pca = transformed_data['X_test_selected']
    y_train = transformed_data['y_train']
    y_test = transformed_data['y_test']
    fruits = getAllFruits()

    # Plot class distribution using Seaborn
    plot_class_distribution_seaborn(y_train, y_test, fruits)

    
if __name__ == "__main__":
    main()