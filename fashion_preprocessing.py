# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.feature import hog
from skimage import exposure
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.transform import resize
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Add the directory containing mnist_reader.py to the path
sys.path.append('./fashion/fashion-mnist/utils')

from mnist_reader import load_mnist

def resize_images(images, new_size=(50, 50)):
    
    resized_images = []
    
    for image in images:
        image_reshaped = image.reshape(28, 28)
        resized_image = resize(image_reshaped, new_size, anti_aliasing=True)
        resized_images.append(resized_image.flatten())
    
    return np.array(resized_images)

def apply_hog(X):
    # Parameters for HOG
    hog_features = []
    hog_images = []

    for image in X:
        image_reshaped = image.reshape(50, 50)
        
        # Compute HOG features
        features, hog_image = hog(image_reshaped, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                                  orientations=9, visualize=True, block_norm='L2-Hys')
        
        # Store HOG features
        hog_features.append(features)
        hog_images.append(hog_image)
    
    # Convert to numpy array
    hog_features = np.array(hog_features)
    hog_images = np.array(hog_images)

    return hog_features, hog_images

def apply_pca(X, n_components=10):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    print(f"Explained variance ratio for {n_components} components: {np.sum(pca.explained_variance_ratio_):.2f}")
    
    return X_pca, pca

def plot_hog_gradients(original_image, hog_features, class_name):
    original_image_reshaped = original_image.reshape(50, 50)

    hog_image = hog_features.reshape(50,50)
    
    # Resize HOG image to match the original image size (28x28)
    hog_image_resized = resize(hog_image, (50, 50), anti_aliasing=True)
    
    # Plot original image and HOG gradients
    ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(original_image_reshaped, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(hog_image_resized, cmap='gray')
    ax[1].set_title('HOG Gradients (Resized)')
    ax[1].axis('off')

    # Overlay original and HOG images
    combined_image = original_image_reshaped / 2 + hog_image_resized / 2
    ax[2].imshow(combined_image, cmap='gray')
    ax[2].set_title('Overlayed Image')
    ax[2].axis('off')

    plt.tight_layout()
    
    plt.savefig('./fashion/pictures_fashion/hog_gradients_overlay_' + str(class_name) + '.png', dpi=500)
    

def main():
    X_train, y_train = load_mnist('./fashion/fashion-mnist/data/fashion', kind='train')
    X_test, y_test = load_mnist('./fashion/fashion-mnist/data/fashion', kind='t10k')
    
    # Save Transformed Data
    np.savez('./fashion/initial_data_fashion.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    
    # Resize X_train and X_test
    X_train_resized = resize_images(X_train, new_size=(50, 50))
    X_test_resized = resize_images(X_test, new_size=(50, 50))
    
    # Save Transformed Data
    np.savez('./fashion/initial_data_fashion_resized.npz', X_train_resized=X_train_resized, X_test_resized=X_test_resized, y_train=y_train, y_test=y_test)
    
    # Apply HOG to training and test images
    X_train_hog, train_hog_images = apply_hog(X_train_resized)
    X_test_hog, test_hog_images = apply_hog(X_test_resized)

    # FOR VISUALIZATION
    for random_class in range(10):
        random_index = np.random.randint(len(X_train_resized[y_train == random_class]))
        original_image = X_train_resized[y_train == random_class][random_index]
        precomputed_hog_features = train_hog_images[y_train == random_class][random_index]

        # Plot HOG gradients over the original image
        plot_hog_gradients(original_image, precomputed_hog_features, random_class)

    # Save Transformed Data
    np.savez('./fashion/hog_data_fashion.npz', X_train_hog=X_train_hog, X_test_hog=X_test_hog, y_train=y_train, y_test=y_test)
    
    # Scale Data Images
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_hog)
    X_test_scaled = scaler.fit_transform(X_test_hog)
    
    np.savez('./fashion/before_selection_data_fashion.npz', X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, y_train=y_train, y_test=y_test)

    selector = SelectPercentile(f_classif, percentile=75)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    np.savez('./fashion/after_selection_data_fashion.npz', X_train_selected=X_train_selected, X_test_selected=X_test_selected, y_train=y_train, y_test=y_test)

    # Apply PCA
    n_components = 50 
    X_train_pca, pca = apply_pca(X_train_selected, n_components=n_components)
    X_test_pca = pca.fit_transform(X_test_selected)
    
    # Save PCA Model
    np.savez('./fashion/pca_model_fashion.npz',
             components=pca.components_, 
             mean=pca.mean_, 
             explained_variance=pca.explained_variance_)

    # Save Transformed Data
    np.savez('./fashion/transformed_data_fashion.npz', X_train_pca=X_train_pca, X_test_pca=X_test_pca, y_train=y_train, y_test=y_test)
    print("Preprocessing completed successfully!")
    
if __name__ == "__main__":
    main()