import numpy as np
import pandas as pd
import cv2
import glob
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from collections import Counter


def normalize_fruit_name(folder_name):
    """
    Normalizes fruit names by using only the first word of the folder name
    to group similar fruits together (e.g., 'Tomato 1', 'Tomato Cherry Red' → 'Tomato').
    """
    return folder_name.split()[0]  # Take the first word as the normalized name

def getAllFruits():
    """
    Gets all unique fruit names from the dataset directory, grouping folders
    with similar prefixes (e.g., 'Tomato 1', 'Tomato Cherry Red') under a single label.
    """
    fruits = []
    path = "./fruits/fruits-360_dataset_100x100/fruits-360/Training/"
    
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            normalized_fruit = normalize_fruit_name(folder)
            if normalized_fruit not in fruits:
                fruits.append(normalized_fruit)
    return fruits

def extract_orb_hog_features(image):
    # Redimensionează imaginea pentru a reduce numărul de descriptori HOG
    resized_image = cv2.resize(image, (64, 64))

    # Extrage descriptorii ORB
    orb = cv2.ORB_create()
    keypoints, orb_descriptors = orb.detectAndCompute(resized_image, None)
    
    # Dacă nu există descriptori ORB (imaginea poate să nu aibă caracteristici distinctive)
    if orb_descriptors is None:
        orb_descriptors = np.zeros((1, 32))  # Set de fallback (vector nul)
    
    # Extrage descriptorii HOG
    hog = cv2.HOGDescriptor(_winSize=(64, 64),  _blockSize=(16, 16),  _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9) 

    hog_descriptors = hog.compute(resized_image)

    # Combină descriptorii ORB și HOG
    features = np.hstack((orb_descriptors.flatten(), hog_descriptors.flatten()))
    
    return features


def getYourFruits(fruits, data_type, print_n=False):
    features = []
    labels = []
    val = ['Training', 'Test']

    path = "./fruits/fruits-360_dataset_100x100/fruits-360/" + data_type + "/"
    for i, fruit in enumerate(fruits):
        # Find all folders that match the current normalized fruit name
        fruit_folders = [
            folder for folder in os.listdir(path)
            if os.path.isdir(os.path.join(path, folder)) and normalize_fruit_name(folder) == fruit
        ]
        j = 0
        for folder in fruit_folders:
            folder_path = os.path.join(path, folder)
            for image_path in glob.glob(os.path.join(folder_path, "*.jpg")):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (100, 100))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extrage caracteristicile ORB + HOG
                image_features = extract_orb_hog_features(image)
                features.append(image_features)
                labels.append(i)
                j += 1
        
        if(print_n):
            print("There are", j, data_type.upper(), "images of", fruits[i].upper())
            
    features = np.array(features)
    labels = np.array(labels)
    return features, labels

def visualize_top_classes(X_train, y_train, fruits):
    # Top 10 clase
    class_counts = Counter(y_train)
    most_common_classes = class_counts.most_common(10)
    top_classes_df = pd.DataFrame(most_common_classes, columns=['Class Index', 'Frequency'])
    top_classes_df['Fruit Name'] = top_classes_df['Class Index'].apply(lambda x: fruits[x])
    top_classes_df = top_classes_df.sort_values(by='Frequency', ascending=False)
    print(top_classes_df)

    # Vizualizare histogramă medie (HOG)
    def calculate_hog_histograms_orb_hog(X_train, y_train, top_classes):
        hog_start_idx = 32  # Index unde încep descriptorii HOG
        histograms = []
        for class_idx in top_classes:
            class_features = X_train[y_train == class_idx, hog_start_idx:]
            mean_histogram = np.mean(class_features, axis=0)
            histograms.append(mean_histogram)
        return histograms

    top_classes = [x[0] for x in most_common_classes]
    hog_histograms = calculate_hog_histograms_orb_hog(X_train, y_train, top_classes)

    plt.figure(figsize=(12, 6))
    for idx, histogram in enumerate(hog_histograms):
        plt.plot(histogram, label=f"{fruits[top_classes[idx]]}")
    plt.xlabel("HOG Feature Index")
    plt.ylabel("Mean Value")
    plt.title("Mean HOG Feature Values for Top Classes")
    plt.legend()
    plt.grid()
    plt.savefig(f'./fruits/pictures_fruits/mean_histogram.png', dpi=300)
    plt.show()

    # Vizualizare puncte cheie ORB pentru o imagine din fiecare clasă
    path = "./fruits/fruits-360_dataset_100x100/fruits-360/Training/"
    for class_idx in top_classes:
        fruit_name = fruits[class_idx]
        fruit_folders = [
            folder for folder in os.listdir(path)
            if os.path.isdir(os.path.join(path, folder)) and normalize_fruit_name(folder) == fruit_name
        ]
        if fruit_folders:
            first_image = glob.glob(os.path.join(path, fruit_folders[0], "*.jpg"))[0]
            plot_orb_keypoints(first_image)
               
def plot_orb_keypoints(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (100, 100))
    orb = cv2.ORB_create()
    keypoints = orb.detect(image, None)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title("ORB Keypoints")
    plt.axis("off")
    plt.show()

def main():
    fruits = getAllFruits()
    
    # Get Images and Labels using ORB + HOG features
    X_train, y_train = getYourFruits(fruits, 'Training', print_n=True)
    X_test, y_test = getYourFruits(fruits, 'Test', print_n=True)

    # Scale Data Images (opțional)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    np.savez('./fruits/before_selection_data_fruits_2.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # Poți continua cu selecția de atribute sau direct cu clasificarea
    selector = SelectPercentile(f_classif, percentile=75)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Save Transformed Data
    np.savez('./fruits/transformed_data_fruits_2.npz', X_train_selected=X_train_selected, X_test_selected=X_test_selected, y_train=y_train, y_test=y_test)
    print("Feature extraction with ORB + HOG completed successfully!")
    
    visualize_top_classes(X_train, y_train, fruits)


if __name__ == "__main__":
    main()