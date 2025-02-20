import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def plot_class_distribution_seaborn(y_train, y_test, class_names):
    # Convert numerical labels back to fruit names
    fruit_names_train = [class_names[label] for label in y_train]
    fruit_names_test = [class_names[label] for label in y_test]

    # Create count plots for training and test sets
    plt.figure(figsize=(20, 10))
    
    # Training set plot
    plt.subplot(1, 1, 1)
    sns.countplot(x=fruit_names_train, order=class_names)
    plt.xticks(rotation=90, fontsize=10)
    plt.title('Training Set Class Distribution')
    plt.xlabel('Fashion Class')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('./fashion/pictures_fashion/training_class_distribution.png', dpi=500)

    plt.figure(figsize=(20, 10))

    # Test set plot
    plt.subplot(1, 1, 1)
    sns.countplot(x=fruit_names_test, order=class_names)
    plt.xticks(rotation=90, fontsize=10)
    plt.title('Test Set Class Distribution')
    plt.xlabel('Fashion Class')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('./fashion/pictures_fashion/test_class_distribution.png', dpi=500)


def showVariance(X_train):
    # Compute covariance matrix using numpy
    cov_matr = np.cov(X_train, rowvar=False)
    eigval, eigvect = np.linalg.eig(cov_matr)
    index = np.argsort(eigval)[::-1] 
    eigvect = eigvect[:, index]
    eigval = eigval[index]

    # Compute explained variance
    n_PC = []
    var_explained = []
    var_temp = []
    var_tmp = 0
    num_components = len(eigval)

    for i in range(num_components):
        var_tmp += eigval[i]
        n_PC.append(i + 1)
        var_temp.append(eigval[i] / eigval.sum() * 100)
        var_explained.append(var_tmp / eigval.sum() * 100)

    # Plotting explained variance
    fig, ax = plt.subplots(figsize=(8, 8))
    ind = np.arange(num_components)
    width = 0.35 
    
    p1 = ax.bar(ind, var_temp, width, color='b')
    p2 = ax.bar(ind + width, var_explained, width, color='r')

    ax.legend((p1[0], p2[0]), ('Individual explained variance', 'Cumulative explained variance'))

    ax.set_title('Variance explained using PCs')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels([str(i + 1) for i in range(num_components)])

    plt.xlabel('Number of PCs')
    plt.ylabel('Variance Explained (%)')

    ax.autoscale_view()
    plt.savefig('./fashion/pictures_fashion/variance.png', dpi=500)
    
def getClassCounts(y):
    _, counts = np.unique(y, return_counts=True)
    return counts

def plotPrincipalComponents(X, y, dim, class_names):
    # Sortează datele pentru a preveni amestecarea claselor
    indices = np.argsort(y)
    X = X[indices]
    y = y[indices]
    
    # Creează o paletă de culori extinsă
    colors = sns.color_palette("tab10", len(class_names))
    markers = ['o', 'x', 'v', 'd']
    
    # Pregătire pentru salvare
    output_dir = './fashion/pictures_fashion/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plotare în funcție de dimensiune
    start = 0
    counts = getClassCounts(y)
    
    if dim == 2:
        for i, count in enumerate(counts):
            end = start + count
            plt.scatter(X[start:end, 0], X[start:end, 1], 
                        color=colors[i % len(colors)], 
                        marker=markers[i % len(markers)], 
                        label=class_names[i])
            start = end
        
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(loc='lower left')
        plt.title('PCA Plot (2D)')
        plt.savefig(output_dir + 'plotprincipalcomponents2.png', dpi=500)
        plt.show()
    
    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for i, count in enumerate(counts):
            end = start + count
            ax.scatter(X[start:end, 0], X[start:end, 1], X[start:end, 2], 
                       color=colors[i % len(colors)], 
                       marker=markers[i % len(markers)], 
                       label=class_names[i])
            start = end
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.1))
        plt.title('PCA Plot (3D)')
        plt.savefig(output_dir + 'plotprincipalcomponents3.png', dpi=500)
        plt.show()

def main():
    transformed_data_fashion = np.load('./fashion/transformed_data_fashion.npz')
    X_train_pca = transformed_data_fashion['X_train_pca']
    X_test_pca = transformed_data_fashion['X_test_pca']
    y_train = transformed_data_fashion['y_train']
    y_test = transformed_data_fashion['y_test']
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    initial_data_fashion = np.load('./fashion/initial_data_fashion.npz')
    X_train_resized = initial_data_fashion['X_train']
    X_test_resized = initial_data_fashion['X_test']
    y_train = initial_data_fashion['y_train']

    pca2 = PCA(n_components=2)
    dataIn2D = pca2.fit_transform(X_train_pca)
    plotPrincipalComponents(dataIn2D, y_train, 2, class_names)
    
    pca3 = PCA(n_components=3)
    dataIn3D = pca3.fit_transform(X_train_pca)
    plotPrincipalComponents(dataIn3D, y_train, 3, class_names)
    
    # Plot class distribution using Seaborn
    plot_class_distribution_seaborn(y_train, y_test, class_names)

    # Show variance explained by each principal component
    showVariance(X_train_pca)

if __name__ == "__main__":
    main()