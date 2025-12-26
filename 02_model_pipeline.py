import argparse
import json
import os
import joblib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix, roc_curve)

warnings.filterwarnings('ignore')

# Runtime configuration
FAST_MODE = False  # when True, uses smaller hyperparameter grids for fast iteration
N_JOBS = -1
DEFAULT_CV = 5

# plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_processed_data(data_path='data/processed/heart_disease_processed.csv'):
    """Load preprocessed data"""
    print("=" * 80)
    print("FEATURE ENGINEERING - HEART DISEASE DATASET")
    print("=" * 80)
    
    print(f"\n[INFO] Loading processed data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  - Shape: {df.shape}")
    
    return df


def prepare_features_and_target(df):
    """
    Separate features and target variable
    """
    print("\n" + "-" * 80)
    print("STEP 1: PREPARING FEATURES AND TARGET")
    print("-" * 80)
    
    # Define target column
    target_col = 'target_binary'
    
    # Exclude columns from features
    exclude_cols = ['target_binary', 'num', 'target', 'condition']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Separate features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"\n[INFO] Features and target prepared:")
    print(f"  - Feature matrix shape: {X.shape}")
    print(f"  - Target vector shape: {y.shape}")
    print(f"\n[INFO] Feature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n[INFO] Target distribution:")
    print(f"  - Class 0 (No Disease): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.2f}%)")
    print(f"  - Class 1 (Disease): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.2f}%)")
    
    return X, y, feature_cols


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    print("\n" + "-" * 80)
    print("STEP 2: TRAIN/TEST SPLIT")
    print("-" * 80)
    
    print(f"\n[INFO] Splitting data with test_size={test_size}, random_state={random_state}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Ensure balanced split
    )
    
    print(f"\n[SUCCESS] Data split completed!")
    print(f"\n  Training set:")
    print(f"    - Samples: {len(X_train)}")
    print(f"    - Class 0: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
    print(f"    - Class 1: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")
    
    print(f"\n  Testing set:")
    print(f"    - Samples: {len(X_test)}")
    print(f"    - Class 0: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)")
    print(f"    - Class 1: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, feature_cols):
    """
    Standardize features using StandardScaler
    Fit on training data, transform both train and test
    """
    print("\n" + "-" * 80)
    print("STEP 3: FEATURE SCALING")
    print("-" * 80)
    
    print(f"\n[INFO] Applying StandardScaler to normalize features...")
    print("  Note: Scaler is fit on training data only to prevent data leakage")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit on training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    print(f"\n[SUCCESS] Features scaled successfully!")
    print(f"\n[INFO] Scaling statistics (training set):")
    print(f"  - Mean of scaled features: ~0.0")
    print(f"  - Std of scaled features: ~1.0")
    
    # Display sample statistics
    print(f"\n[INFO] Sample feature statistics after scaling:")
    print(f"{'Feature':<15} {'Train Mean':>12} {'Train Std':>12} {'Test Mean':>12} {'Test Std':>12}")
    print("-" * 65)
    for col in feature_cols[:5]:  # Show first 5 features
        print(f"{col:<15} {X_train_scaled[col].mean():>12.4f} {X_train_scaled[col].std():>12.4f} "
              f"{X_test_scaled[col].mean():>12.4f} {X_test_scaled[col].std():>12.4f}")
    if len(feature_cols) > 5:
        print(f"{'...':<15} {'...':>12} {'...':>12} {'...':>12} {'...':>12}")
    
    return X_train_scaled, X_test_scaled, scaler


def save_engineered_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler, 
                        feature_cols, output_dir='data/engineered'):
    """
    Save engineered features and scaler
    """
    print("\n" + "-" * 80)
    print("STEP 4: SAVING ENGINEERED DATA")
    print("-" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train/test splits
    print("\n[INFO] Saving train/test datasets...")
    
    # Combine features and target for saving
    train_data = X_train_scaled.copy()
    train_data['target'] = y_train.values
    
    test_data = X_test_scaled.copy()
    test_data['target'] = y_test.values
    
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"  ✓ Training data saved: {train_path}")
    print(f"  ✓ Testing data saved: {test_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Scaler saved: {scaler_path}")
    
    # Save feature names
    feature_names_path = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"  ✓ Feature names saved: {feature_names_path}")
    
    # Create metadata file
    metadata = {
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train_scaled),
        'n_test_samples': len(X_test_scaled),
        'train_class_0': int((y_train == 0).sum()),
        'train_class_1': int((y_train == 1).sum()),
        'test_class_0': int((y_test == 0).sum()),
        'test_class_1': int((y_test == 1).sum()),
        'test_size': 0.2,
        'random_state': 42,
        'scaling_method': 'StandardScaler'
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"  ✓ Metadata saved: {metadata_path}")
    
    print(f"\n[SUCCESS] All engineered data saved to: {output_dir}")
    
    return {
        'train_path': train_path,
        'test_path': test_path,
        'scaler_path': scaler_path,
        'feature_names_path': feature_names_path,
        'metadata_path': metadata_path
    }


def feature_engineering_pipeline(data_path='data/processed/heart_disease_processed.csv',
                                  output_dir='data/engineered',
                                  test_size=0.2,
                                  random_state=42):
    """
    Complete feature engineering pipeline
    """
    # Load data
    df = load_processed_data(data_path)
    
    # Prepare features and target
    X, y, feature_cols = prepare_features_and_target(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, feature_cols)
    
    # Save engineered data
    paths = save_engineered_data(X_train_scaled, X_test_scaled, y_train, y_test, 
                                 scaler, feature_cols, output_dir)
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nData is now ready for model training!")
    print(f"\nGenerated files in '{output_dir}':")
    for key, path in paths.items():
        print(f"  - {key}: {os.path.basename(path)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

"""
Model Training Script for Heart Disease UCI Dataset
Trains multiple classification models with hyperparameter tuning
"""


def load_engineered_data(data_dir='data/engineered'):
    """Load engineered train/test data"""
    print("=" * 80)
    print("MODEL TRAINING - HEART DISEASE DATASET")
    print("=" * 80)
    
    print(f"\n[INFO] Loading engineered data from: {data_dir}")
    
    train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    
    # Separate features and target
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    print(f"  ✓ Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  ✓ Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, cv=5):
    """
    Train Logistic Regression with hyperparameter tuning
    """
    print("\n" + "=" * 80)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("=" * 80)
    
    print("\n[INFO] Configuring hyperparameter grid...")
    if FAST_MODE:
        param_grid = {
            'C': [0.1, 1],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'max_iter': [500]
        }
    else:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000]
        }
    
    print(f"  - Parameter combinations to test: {len(param_grid['C']) * len(param_grid['solver'])}")
    print(f"  - Cross-validation folds: {cv}")
    
    print("\n[INFO] Training Logistic Regression with Grid Search CV...")
    
    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        lr, param_grid, cv=cv, scoring='roc_auc', 
        n_jobs=N_JOBS, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n[SUCCESS] Training completed!")
    print(f"  - Best parameters: {grid_search.best_params_}")
    print(f"  - Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_random_forest(X_train, y_train, cv=5):
    """
    Train Random Forest with hyperparameter tuning
    """
    print("\n" + "=" * 80)
    print("MODEL 2: RANDOM FOREST")
    print("=" * 80)
    
    print("\n[INFO] Configuring hyperparameter grid...")
    if FAST_MODE:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt']
        }
    else:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    
    total_combinations = (len(param_grid['n_estimators']) * 
                         len(param_grid['max_depth']) * 
                         len(param_grid['min_samples_split']) * 
                         len(param_grid['min_samples_leaf']) * 
                         len(param_grid['max_features']))
    
    print(f"  - Parameter combinations to test: {total_combinations}")
    print(f"  - Cross-validation folds: {cv}")
    print("  - Note: This may take a few minutes...")
    
    print("\n[INFO] Training Random Forest with Grid Search CV...")
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring='roc_auc', 
        n_jobs=N_JOBS, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n[SUCCESS] Training completed!")
    print(f"  - Best parameters: {grid_search.best_params_}")
    print(f"  - Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_svm(X_train, y_train, cv=5):
    """
    Train Support Vector Machine with hyperparameter tuning
    """
    print("\n" + "=" * 80)
    print("MODEL 3: SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 80)
    
    print("\n[INFO] Configuring hyperparameter grid...")
    if FAST_MODE:
        param_grid = {
            'C': [0.1, 1],
            'kernel': ['rbf'],
            'gamma': ['scale', 0.001]
        }
    else:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        }
    
    total_combinations = (len(param_grid['C']) * 
                         len(param_grid['kernel']) * 
                         len(param_grid['gamma']))
    
    print(f"  - Parameter combinations to test: {total_combinations}")
    print(f"  - Cross-validation folds: {cv}")
    
    print("\n[INFO] Training SVM with Grid Search CV...")
    
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid, cv=cv, scoring='roc_auc', 
        n_jobs=N_JOBS, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n[SUCCESS] Training completed!")
    print(f"  - Best parameters: {grid_search.best_params_}")
    print(f"  - Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_gradient_boosting(X_train, y_train, cv=5):
    """
    Train Gradient Boosting with hyperparameter tuning
    """
    print("\n" + "=" * 80)
    print("MODEL 4: GRADIENT BOOSTING")
    print("=" * 80)
    
    print("\n[INFO] Configuring hyperparameter grid...")
    if FAST_MODE:
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    else:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    total_combinations = (len(param_grid['n_estimators']) * 
                         len(param_grid['learning_rate']) * 
                         len(param_grid['max_depth']) * 
                         len(param_grid['min_samples_split']) * 
                         len(param_grid['min_samples_leaf']))
    
    print(f"  - Parameter combinations to test: {total_combinations}")
    print(f"  - Cross-validation folds: {cv}")
    print("  - Note: This may take a few minutes...")
    
    print("\n[INFO] Training Gradient Boosting with Grid Search CV...")
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(
        gb, param_grid, cv=cv, scoring='roc_auc', 
        n_jobs=N_JOBS, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n[SUCCESS] Training completed!")
    print(f"  - Best parameters: {grid_search.best_params_}")
    print(f"  - Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def save_model(model, model_name, best_params, cv_score, output_dir='models'):
    """Save trained model and metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_name}.pkl')
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'best_params': best_params,
        'cv_roc_auc_score': float(cv_score),
        'model_type': type(model).__name__
    }
    
    metadata_path = os.path.join(output_dir, f'{model_name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"  ✓ Model saved: {model_path}")
    print(f"  ✓ Metadata saved: {metadata_path}")
    
    return model_path, metadata_path


def train_all_models(X_train, y_train, cv=5, output_dir='models'):
    """
    Train all classification models
    """
    models = {}
    
    # Logistic Regression
    lr_model, lr_params, lr_score = train_logistic_regression(X_train, y_train, cv)
    save_model(lr_model, 'logistic_regression', lr_params, lr_score, output_dir)
    models['logistic_regression'] = {
        'model': lr_model,
        'params': lr_params,
        'cv_score': lr_score
    }
    
    # Random Forest
    rf_model, rf_params, rf_score = train_random_forest(X_train, y_train, cv)
    save_model(rf_model, 'random_forest', rf_params, rf_score, output_dir)
    models['random_forest'] = {
        'model': rf_model,
        'params': rf_params,
        'cv_score': rf_score
    }
    
    # SVM
    svm_model, svm_params, svm_score = train_svm(X_train, y_train, cv)
    save_model(svm_model, 'svm', svm_params, svm_score, output_dir)
    models['svm'] = {
        'model': svm_model,
        'params': svm_params,
        'cv_score': svm_score
    }
    
    # Gradient Boosting
    gb_model, gb_params, gb_score = train_gradient_boosting(X_train, y_train, cv)
    save_model(gb_model, 'gradient_boosting', gb_params, gb_score, output_dir)
    models['gradient_boosting'] = {
        'model': gb_model,
        'params': gb_params,
        'cv_score': gb_score
    }
    
    return models


def model_training_pipeline(data_dir='data/engineered', 
                            output_dir='models',
                            cv=DEFAULT_CV):
    """
    Complete model training pipeline
    """
    # Load data
    X_train, X_test, y_train, y_test = load_engineered_data(data_dir)
    
    # Train all models
    print("\n" + "=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)
    print("\nThis will train 4 models with hyperparameter tuning.")
    print("Please wait while training completes...\n")
    
    models = train_all_models(X_train, y_train, cv, output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETED")
    print("=" * 80)
    
    print("\n[INFO] Training Summary:")
    print(f"\n{'Model':<25} {'CV ROC-AUC':>15}")
    print("-" * 42)
    for model_name, model_info in models.items():
        print(f"{model_name.replace('_', ' ').title():<25} {model_info['cv_score']:>15.4f}")
    
    # Find best model
    best_model_name = max(models.items(), key=lambda x: x[1]['cv_score'])[0]
    best_score = models[best_model_name]['cv_score']
    
    print(f"\n[SUCCESS] Best Model: {best_model_name.replace('_', ' ').title()}")
    print(f"           CV ROC-AUC: {best_score:.4f}")
    
    print(f"\n[INFO] All models saved to: {output_dir}/")
    
    return models, X_train, X_test, y_train, y_test


"""
Model Evaluation Script for Heart Disease UCI Dataset
Evaluates all trained models with comprehensive metrics and visualizations
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix, 
    classification_report, precision_recall_curve, auc
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data_and_models(data_dir='data/engineered', models_dir='models'):
    """Load test data and all trained models"""
    print("=" * 80)
    print("MODEL EVALUATION - HEART DISEASE DATASET")
    print("=" * 80)
    
    # Load test data
    print(f"\n[INFO] Loading test data from: {data_dir}")
    test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    print(f"  ✓ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Load models
    print(f"\n[INFO] Loading trained models from: {models_dir}")
    models = {}
    # Ensure custom preprocessor class is available for unpickling
    try:
        import sys
        from preprocessing_pipeline import HeartDiseasePreprocessor
        setattr(sys.modules['__main__'], 'HeartDiseasePreprocessor', HeartDiseasePreprocessor)
    except Exception:
        # If import fails, continue and let joblib raise a clear error when loading
        pass
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    for model_file in model_files:
        model_name = model_file.replace('.pkl', '')
        model_path = os.path.join(models_dir, model_file)
        loaded_obj = joblib.load(model_path)
        # Only include objects that behave like estimators (have predict & predict_proba)
        if not (hasattr(loaded_obj, 'predict') and hasattr(loaded_obj, 'predict_proba')):
            print(f"  - Skipping non-model pickle: {model_file}")
            continue
        models[model_name] = loaded_obj
        print(f"  ✓ Loaded: {model_name}")
    
    print(f"\n[SUCCESS] Loaded {len(models)} models for evaluation")
    
    return X_test, y_test, models


def evaluate_model(model, model_name, X_test, y_test):
    """
    Evaluate a single model with comprehensive metrics
    """
    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {model_name.upper().replace('_', ' ')}")
    print('=' * 80)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Display metrics
    print(f"\n[METRICS]")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n[CONFUSION MATRIX]")
    print(f"                 Predicted")
    print(f"                 No  Yes")
    print(f"Actual  No      {cm[0,0]:3d}  {cm[0,1]:3d}")
    print(f"        Yes     {cm[1,0]:3d}  {cm[1,1]:3d}")
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    print(f"\n[ADDITIONAL METRICS]")
    print(f"  Sensitivity (TPR): {sensitivity:.4f}")
    print(f"  Specificity (TNR): {specificity:.4f}")
    print(f"  False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"  False Negative Rate: {fn/(fn+tp):.4f}")
    
    # Classification Report
    print(f"\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
    
    # Store results
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist()
    }
    
    return results


def plot_confusion_matrices(all_results, output_dir):
    """Plot confusion matrices for all models"""
    print(f"\n{'=' * 80}")
    print("GENERATING CONFUSION MATRIX VISUALIZATIONS")
    print('=' * 80)
    
    n_models = len(all_results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        cm = np.array(results['confusion_matrix'])
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'],
                   ax=axes[idx], cbar_kws={'label': 'Count'})
        
        axes[idx].set_title(f'{model_name.replace("_", " ").title()}\n'
                           f'Accuracy: {results["accuracy"]:.4f}',
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Actual', fontsize=11)
        axes[idx].set_xlabel('Predicted', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '05_confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Confusion matrices saved: {output_path}")


def plot_roc_curves(all_results, y_test, output_dir):
    """Plot ROC curves for all models"""
    print(f"\n[INFO] Generating ROC curves...")
    
    plt.figure(figsize=(10, 8))
    
    for model_name, results in all_results.items():
        y_pred_proba = np.array(results['probabilities'])
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = results['roc_auc']
        
        plt.plot(fpr, tpr, linewidth=2, 
                label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.4f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    output_path = os.path.join(output_dir, '06_roc_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ ROC curves saved: {output_path}")


def plot_metrics_comparison(all_results, output_dir):
    """Create bar chart comparing all metrics across models"""
    print(f"\n[INFO] Generating metrics comparison chart...")
    
    # Prepare data
    models = []
    metrics_data = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'ROC-AUC': []
    }
    
    for model_name, results in all_results.items():
        models.append(model_name.replace('_', ' ').title())
        metrics_data['Accuracy'].append(results['accuracy'])
        metrics_data['Precision'].append(results['precision'])
        metrics_data['Recall'].append(results['recall'])
        metrics_data['F1-Score'].append(results['f1_score'])
        metrics_data['ROC-AUC'].append(results['roc_auc'])
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # Individual metric plots
    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        ax = axes[idx]
        bars = ax.bar(range(len(models)), values, color=[colors[i] for i in range(len(models))], 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Overall comparison (last subplot)
    ax = axes[5]
    x = np.arange(len(models))
    width = 0.15
    
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        offset = width * (i - 2)
        ax.bar(x + offset, values, width, label=metric_name, alpha=0.8, edgecolor='black')
    
    ax.set_title('All Metrics Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '07_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Metrics comparison saved: {output_path}")


def save_evaluation_results(all_results, output_dir):
    """Save evaluation results to files"""
    print(f"\n[INFO] Saving evaluation results...")
    
    # Save detailed results as JSON
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    
    # Convert numpy types to native Python types for JSON serialization
    json_results = {}
    for model_name, results in all_results.items():
        json_results[model_name] = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'roc_auc': float(results['roc_auc']),
            'sensitivity': float(results['sensitivity']),
            'specificity': float(results['specificity']),
            'confusion_matrix': results['confusion_matrix']
        }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    print(f"  ✓ JSON results saved: {results_path}")
    
    # Save summary as CSV
    summary_df = pd.DataFrame({
        'Model': [name.replace('_', ' ').title() for name in all_results.keys()],
        'Accuracy': [r['accuracy'] for r in all_results.values()],
        'Precision': [r['precision'] for r in all_results.values()],
        'Recall': [r['recall'] for r in all_results.values()],
        'F1-Score': [r['f1_score'] for r in all_results.values()],
        'ROC-AUC': [r['roc_auc'] for r in all_results.values()],
        'Sensitivity': [r['sensitivity'] for r in all_results.values()],
        'Specificity': [r['specificity'] for r in all_results.values()]
    })
    
    summary_path = os.path.join(output_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ CSV summary saved: {summary_path}")
    
    return summary_df


def model_evaluation_pipeline(data_dir='data/engineered',
                              models_dir='models',
                              output_dir='outputs/model_evaluation'):
    """
    Complete model evaluation pipeline
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data and models
    X_test, y_test, models = load_data_and_models(data_dir, models_dir)
    
    # Evaluate all models
    all_results = {}
    for model_name, model in models.items():
        results = evaluate_model(model, model_name, X_test, y_test)
        all_results[model_name] = results
    
    # Generate visualizations
    print(f"\n{'=' * 80}")
    print("GENERATING VISUALIZATIONS")
    print('=' * 80)
    
    plot_confusion_matrices(all_results, output_dir)
    plot_roc_curves(all_results, y_test, output_dir)
    plot_metrics_comparison(all_results, output_dir)
    
    # Save results
    summary_df = save_evaluation_results(all_results, output_dir)
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("MODEL EVALUATION COMPLETED")
    print('=' * 80)
    
    print(f"\n[SUMMARY TABLE]")
    print(summary_df.to_string(index=False))
    
    # Find best model
    best_model = summary_df.loc[summary_df['F1-Score'].idxmax()]
    
    print(f"\n{'=' * 80}")
    print(f"🏆 BEST MODEL: {best_model['Model']}")
    print('=' * 80)
    print(f"  Accuracy:    {best_model['Accuracy']:.4f}")
    print(f"  Precision:   {best_model['Precision']:.4f}")
    print(f"  Recall:      {best_model['Recall']:.4f}")
    print(f"  F1-Score:    {best_model['F1-Score']:.4f}")
    print(f"  ROC-AUC:     {best_model['ROC-AUC']:.4f}")
    
    print(f"\n[INFO] All evaluation results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. 05_confusion_matrices.png - Confusion matrices for all models")
    print("  2. 06_roc_curves.png - ROC curves comparison")
    print("  3. 07_metrics_comparison.png - Metrics comparison charts")
    print("  4. evaluation_results.json - Detailed results in JSON format")
    print("  5. evaluation_summary.csv - Summary table in CSV format")
    
    return all_results, summary_df



if __name__ == "__main__":
    # If no CLI args provided, run full pipeline (feature engineering, training, eval)
    X_train, X_test, y_train, y_test, scaler, feature_cols = feature_engineering_pipeline()
    print(f"\n[INFO] Ready for model training with {len(feature_cols)} features!")
    models, X_train, X_test, y_train, y_test = model_training_pipeline()
    print("\n[INFO] Models are ready for evaluation!")
    all_results, summary_df = model_evaluation_pipeline()
    print("\n[SUCCESS] Model evaluation complete!")
    print("          Review the visualizations and metrics to select the best model.")
