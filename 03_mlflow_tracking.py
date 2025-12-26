"""
MLflow Experiment Tracking for Heart Disease Prediction
Integrates MLflow to log parameters, metrics, and artifacts for all model experiments
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set experiment name
EXPERIMENT_NAME = "heart_disease_classification"


def setup_mlflow_experiment():
    """Setup MLflow experiment"""
    print("=" * 80)
    print("MLFLOW EXPERIMENT TRACKING SETUP")
    print("=" * 80)

    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location="./mlflow_artifacts"
        )
        print(f"\n[INFO] Created new experiment: {EXPERIMENT_NAME}")
    except:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id
        print(f"\n[INFO] Using existing experiment: {EXPERIMENT_NAME}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"  - Experiment ID: {experiment_id}")
    print(f"  - Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"  - Artifact Location: ./mlflow_artifacts")

    return experiment_id


def load_data(data_dir="data/engineered"):
    """Load engineered train/test data"""
    print(f"\n[INFO] Loading data from: {data_dir}")

    train_data = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "test_data.csv"))

    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]
    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    print(f"  ✓ Training: {X_train.shape[0]} samples")
    print(f"  ✓ Testing: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Generate confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()

    # Save plot
    plot_path = f"temp_{title.replace(' ', '_').lower()}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """Generate ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_path = f"temp_{title.replace(' ', '_').lower()}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def train_and_log_model(model_name, model, param_grid, X_train, y_train, X_test, y_test, cv=5):
    """
    Train model with GridSearchCV and log everything to MLflow
    """
    print(f"\n{'=' * 80}")
    print(f"TRAINING AND LOGGING: {model_name.upper()}")
    print("=" * 80)

    # Start MLflow run
    with mlflow.start_run(run_name=model_name):

        # Log basic info
        mlflow.set_tag("model_type", type(model).__name__)
        mlflow.set_tag("training_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Log dataset info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("cv_folds", cv)

        # GridSearchCV
        print(f"\n[INFO] Training with GridSearchCV...")
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_

        print(f"\n[SUCCESS] Training completed!")
        print(f"  - Best params: {best_params}")
        print(f"  - CV ROC-AUC: {cv_score:.4f}")

        # Log hyperparameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # Log CV score
        mlflow.log_metric("cv_roc_auc", cv_score)

        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            # Training metrics
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred),
            "train_recall": recall_score(y_train, y_train_pred),
            "train_f1": f1_score(y_train, y_train_pred),
            "train_roc_auc": roc_auc_score(y_train, y_train_proba),
            # Test metrics
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "test_f1": f1_score(y_test, y_test_pred),
            "test_roc_auc": roc_auc_score(y_test, y_test_proba),
        }

        # Log all metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        print(f"\n[METRICS]")
        print(
            f"  Training - Accuracy: {metrics['train_accuracy']:.4f}, ROC-AUC: {metrics['train_roc_auc']:.4f}"
        )
        print(
            f"  Testing  - Accuracy: {metrics['test_accuracy']:.4f}, ROC-AUC: {metrics['test_roc_auc']:.4f}"
        )

        # Generate and log plots
        print(f"\n[INFO] Generating and logging plots...")

        # Confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_test_pred, f"{model_name} Confusion Matrix")
        mlflow.log_artifact(cm_path, "plots")
        os.remove(cm_path)

        # ROC curve
        roc_path = plot_roc_curve(y_test, y_test_proba, f"{model_name} ROC Curve")
        mlflow.log_artifact(roc_path, "plots")
        os.remove(roc_path)

        print(f"  ✓ Confusion matrix logged")
        print(f"  ✓ ROC curve logged")

        # Log model with signature
        signature = infer_signature(X_train, y_train_pred)
        mlflow.sklearn.log_model(
            best_model,
            "model",
            signature=signature,
            registered_model_name=f"heart_disease_{model_name}",
        )
        print(f"  ✓ Model logged")

        # Log additional artifacts
        model_info = {
            "model_name": model_name,
            "model_type": type(best_model).__name__,
            "best_params": best_params,
            "cv_score": cv_score,
            "metrics": metrics,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        info_path = "temp_model_info.json"
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=4)
        mlflow.log_artifact(info_path, "metadata")
        os.remove(info_path)

        print(f"  ✓ Metadata logged")
        print(f"\n[SUCCESS] MLflow run completed for {model_name}")

        return best_model, metrics


def run_all_mlflow_experiments(data_dir="data/engineered", cv=5):
    """
    Run all model experiments with MLflow tracking
    """
    # Setup MLflow
    experiment_id = setup_mlflow_experiment()

    # Load data
    X_train, X_test, y_train, y_test = load_data(data_dir)

    # Define models and parameter grids
    experiments = {
        "logistic_regression": {
            "model": LogisticRegression(random_state=42),
            "params": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"],
                "max_iter": [1000],
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        },
        "svm": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto", 0.001, 0.01],
            },
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
    }

    # Train and log all models
    print(f"\n{'=' * 80}")
    print("RUNNING ALL MLFLOW EXPERIMENTS")
    print("=" * 80)
    print(f"\nTotal experiments to run: {len(experiments)}")

    results = {}
    for exp_name, exp_config in experiments.items():
        model, metrics = train_and_log_model(
            exp_name,
            exp_config["model"],
            exp_config["params"],
            X_train,
            y_train,
            X_test,
            y_test,
            cv=cv,
        )
        results[exp_name] = {"model": model, "metrics": metrics}

    # Print summary
    print(f"\n{'=' * 80}")
    print("ALL MLFLOW EXPERIMENTS COMPLETED")
    print("=" * 80)

    print(f"\n[SUMMARY]")
    print(f"\n{'Model':<25} {'Test Accuracy':>15} {'Test ROC-AUC':>15}")
    print("-" * 57)
    for model_name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{model_name.replace('_', ' ').title():<25} "
            f"{metrics['test_accuracy']:>15.4f} "
            f"{metrics['test_roc_auc']:>15.4f}"
        )

    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]["metrics"]["test_roc_auc"])[0]
    best_roc_auc = results[best_model_name]["metrics"]["test_roc_auc"]

    print(f"\n[BEST MODEL] {best_model_name.replace('_', ' ').title()}")
    print(f"             Test ROC-AUC: {best_roc_auc:.4f}")

    print(f"\n[INFO] MLflow Tracking:")
    print(f"  - Experiment: {EXPERIMENT_NAME}")
    print(f"  - Runs logged: {len(experiments)}")
    print(f"  - Tracking URI: {MLFLOW_TRACKING_URI}")

    print(f"\n[INFO] To view experiments, run:")
    print(f"       mlflow ui")
    print(f"       Then open: http://127.0.0.1:5000")

    return results


if __name__ == "__main__":
    # Run all experiments with MLflow tracking
    results = run_all_mlflow_experiments()

    print("\n[SUCCESS] All experiments logged to MLflow!")
    print("          Launch MLflow UI to compare runs and analyze results.")
