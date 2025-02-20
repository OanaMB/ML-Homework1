from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

# Define the classifiers
log_reg = LogisticRegression(max_iter=10000)
svm = SVC()
rf = RandomForestClassifier(n_jobs=16)
xgboost_model = xgb.XGBClassifier(n_jobs=16)

# Hyperparameters to tune
param_dist_log_reg = {
    'C': uniform(loc=0, scale=4),  # Regularization parameter
    'multi_class': ['ovr', 'multinomial'],  # Classification strategy
}

param_dist_svm = {
    'C': uniform(loc=0, scale=4),  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel types
}

param_dist_rf = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [10, 20, 30, None],  # Max depth of trees
    'max_features': ['auto', 'sqrt', 'log2'],  # Features per tree
}

param_dist_xgb = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [3, 5, 7, 9],  # Max depth of trees
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate
}

# Apply RandomizedSearchCV for each model
def perform_randomized_search(model, param_dist, X_train, y_train):
    randomized_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5)
    randomized_search.fit(X_train, y_train)
    return randomized_search.best_params_, randomized_search.best_score_

def evaluate_model(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    per_class_metrics = {
        class_name: {
            "Precision": round(metrics["precision"], 3),
            "Recall": round(metrics["recall"], 3),
            "F1-Score": round(metrics["f1-score"], 3),
        }
        for class_name, metrics in report.items() if class_name.isdigit()
    }
    
    # Create a structured table
    result = {"Model": model_name, "Accuracy": round(accuracy, 3)}
    for class_name, metrics in per_class_metrics.items():
        for metric_name, value in metrics.items():
            result[f"Class {class_name} {metric_name}"] = value
    return result

def plot_confusion_matrix(y_test, y_pred, classes, model_name, dataset_name):
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Display matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix - {model_name} ({dataset_name})")
    plt.savefig(f'./fashion/pictures_fashion/{model_name}_confusion_matrix.png', dpi=500)
    plt.show()

def main():
    transformed_data_fashion = np.load('./fashion/transformed_data_fashion.npz')
    X_train_pca = transformed_data_fashion['X_train_pca']
    X_test_pca = transformed_data_fashion['X_test_pca']
    y_train = transformed_data_fashion['y_train']
    y_test = transformed_data_fashion['y_test']

    best_params_log_reg, best_score_log_reg = perform_randomized_search(log_reg, param_dist_log_reg, X_train_pca, y_train)
    best_params_svm, best_score_svm = perform_randomized_search(svm, param_dist_svm, X_train_pca, y_train)
    best_params_rf, best_score_rf = perform_randomized_search(rf, param_dist_rf, X_train_pca, y_train)
    best_params_xgb, best_score_xgb = perform_randomized_search(xgboost_model, param_dist_xgb, X_train_pca, y_train)

    print(f"Best Logistic Regression Parameters: {best_params_log_reg}, Best Score: {best_score_log_reg}")
    print(f"Best SVM Parameters: {best_params_svm}, Best Score: {best_score_svm}")
    print(f"Best Random Forest Parameters: {best_params_rf}, Best Score: {best_score_rf}")
    print(f"Best XGBoost Parameters: {best_params_xgb}, Best Score: {best_score_xgb}")

    # Train the models with the best parameters
    log_reg_best = LogisticRegression(**best_params_log_reg)
    svm_best = SVC(**best_params_svm)
    rf_best = RandomForestClassifier(**best_params_rf)
    xgb_best = xgb.XGBClassifier(**best_params_xgb)

    # Antrenare modele
    models = {
        "Logistic Regression": log_reg_best,
        "SVM": svm_best,
        "Random Forest": rf_best,
        "XGBoost": xgb_best
    }
    
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    results = []
    for model_name, model in models.items():
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        result = evaluate_model(y_test, y_pred, model_name)
        
        plot_confusion_matrix(y_test, y_pred, classes, model_name, dataset_name="PCA Transformed Data")
        results.append(result)
        
    scaled_data_fashion = np.load('./fashion/before_selection_data_fashion.npz')
    X_train_scaled = scaled_data_fashion['X_train_scaled']
    total_features_before = X_train_scaled.shape[1]
    print(f"Total features before selection: {total_features_before}")
    
    scaled_data_fashion = np.load('./fashion/after_selection_data_fashion.npz')
    X_train_selected = scaled_data_fashion['X_train_selected']
    total_features_before = X_train_selected.shape[1]
    print(f"Total features before selection: {total_features_before}")
    
    # Creăm tabelul final
    df_results = pd.DataFrame(results)
    df_results.to_csv('./fashion/results.csv')
    print(df_results)
    
    # Evidențiere valori maxime
    def highlight_max(s):
        is_max = s == s.max()
        return ['font-weight: bold' if v else '' for v in is_max]

    styled_results = df_results.style.apply(
        highlight_max, subset=df_results.columns[1:], axis=0
    )
    
    # Salvează DataFrame-ul stilizat ca fișier HTML
    styled_results.to_html("./fashion/styled_results.html")
    print("Styled results saved as styled_results.html")


if __name__ == '__main__':
    main()