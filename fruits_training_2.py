from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
from fruits_preprocessing_2 import getAllFruits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import time
import os

# Define the classifiers and their hyperparameters
param_grid_log_reg = {
    'C': [0.1, 1, 10],
    'multi_class': ['ovr', 'multinomial'],
}

param_grid_svm = {
    'C': [0.1, 100],
    'kernel': ['linear', 'rbf'],
}

param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'max_features': ['sqrt', 'log2'],
}

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
}

# Generate combinations of hyperparameters
def generate_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    for combination in product(*values):
        yield dict(zip(keys, combination))

# Function to evaluate model performance
def evaluate_model(y_test, y_pred, model_name):
    class_names = getAllFruits()
    
    # Calculate the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Extract metrics for macro avg and weighted avg
    macro_avg = {
        "Precision": round(report["macro avg"]["precision"], 3),
        "Recall": round(report["macro avg"]["recall"], 3),
        "F1-Score": round(report["macro avg"]["f1-score"], 3),
    }
    weighted_avg = {
        "Precision": round(report["weighted avg"]["precision"], 3),
        "Recall": round(report["weighted avg"]["recall"], 3),
        "F1-Score": round(report["weighted avg"]["f1-score"], 3),
    }
    
    # Extract metrics per class
    per_class_metrics = {
        class_name: {
            "Precision": round(metrics["precision"], 3),
            "Recall": round(metrics["recall"], 3),
            "F1-Score": round(metrics["f1-score"], 3),
        }
        for class_name, metrics in report.items() if class_name in class_names
    }
    
    # Create a structured result
    result = {"Model": model_name, "Accuracy": round(accuracy, 3)}
    for class_name, metrics in per_class_metrics.items():
        for metric_name, value in metrics.items():
            result[f"{class_name} {metric_name}"] = value
    
    # Add macro avg and weighted avg to the results
    for metric_name, value in macro_avg.items():
        result[f"Macro Avg {metric_name}"] = value
    for metric_name, value in weighted_avg.items():
        result[f"Weighted Avg {metric_name}"] = value
    
    return result


def plot_confusion_matrix(y_val, y_pred, model_name, dataset_name, save_path="./fruits/pictures_fruits"):
    # Get the list of fruits and the top 10 most common classes in the actual labels (y_val)
    fruits = getAllFruits()
    class_counts = Counter(y_val)
    most_common_classes = class_counts.most_common(10)
    
    # Get the class indices for the top 10 classes
    top_classes = [x[0] for x in most_common_classes]
    
    # Filter out the predictions and true labels that belong to these top classes
    y_val_top = [label for label in y_val if label in top_classes]
    y_pred_top = [pred for label, pred in zip(y_val, y_pred) if label in top_classes]
    
    # Create a mapping from class indices to fruit names
    top_classes_df = pd.DataFrame(most_common_classes, columns=['Class Index', 'Frequency'])
    top_classes_df['Fruit Name'] = top_classes_df['Class Index'].apply(lambda x: fruits[x])
    top_classes_df = top_classes_df.sort_values(by='Frequency', ascending=False)
    
    # Create the confusion matrix for the top 10 classes
    cm = confusion_matrix(y_val_top, y_pred_top, labels=top_classes)
    
    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=top_classes_df['Fruit Name'].values)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    
    # Create the save path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save the plot
    plt.title(f"{model_name} Confusion Matrix ({dataset_name})")
    plt.savefig(f"{save_path}/{model_name}_confusion_matrix.png", dpi=300)
    plt.close()

# Main function
def main():
    st = time.time()

    # Load dataset
    data = np.load('./fruits/transformed_data_fruits_2.npz')
    X_train_full, X_test, y_train_full, y_test = (
        data['X_train_selected'],
        data['X_test_selected'],
        data['y_train'],
        data['y_test'],
    )

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    print(f"Train data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    classifiers = {
        "Random Forest": (RandomForestClassifier(n_jobs = -1), param_grid_rf),
        "Logistic Regression": (LogisticRegression(max_iter=5000), param_grid_log_reg),
        "XGBoost": (xgb.XGBClassifier(n_jobs = -1), param_grid_xgb),
        "SVM": (SVC(), param_grid_svm),
    }

    results = []
    for model_name, (model, param_grid) in classifiers.items():
        print(f"Tuning {model_name}...")
        best_accuracy = 0
        best_params = None

        for params in generate_combinations(param_grid):
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            accuracy = accuracy_score(y_val, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params

        print(f"Best parameters for {model_name}: {best_params} with accuracy {best_accuracy:.3f}")

        # Train with best parameters
        best_model = model.set_params(**best_params)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
         # Sanitize file name
        params_str = "_".join(f"{key}={value}" for key, value in best_params.items())
        name = f"{model_name}_{params_str}"

        result = evaluate_model(y_test, y_pred, name)
        results.append(result)

        
        plot_confusion_matrix(y_test, y_pred, name, "HOG+ORB", save_path="./fruits/pictures_fruits")

    # Save results
    df_results = pd.DataFrame(results)
    df_results = df_results.transpose()
    df_results.to_csv('./fruits/final_results.csv', index=False)
    
    
    # Highlight the maximum value for each row
    def highlight_max_row(row):
        is_max = row == row.max()
        return ['font-weight: bold' if v else '' for v in is_max]

    # Apply row-wise highlighting to the styled DataFrame
    styled_results = df_results.style.apply(
        highlight_max_row, axis=1
    )
    styled_results.to_html("./fruits/styled_results.html")
    
    scaled_data_fruits = np.load('./fruits/before_selection_data_fruits_2.npz')
    X_train_scaled = scaled_data_fruits['X_train']
    total_features_before = X_train_scaled.shape[1]
    print(f"Total features before selection: {total_features_before}")
    
    total_features_after = X_train_full.shape[1]
    print(f"Total features after selection: {total_features_after}")

    et = time.time()
    print(f"Execution time: {(et - st) / 60:.2f} minutes")


if __name__ == '__main__':
    main()
