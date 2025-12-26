import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional heavy deps imported only when needed
try:
    from ucimlrepo import fetch_ucirepo
except Exception:
    fetch_ucirepo = None

try:
    import sweetviz as sv
except Exception:
    sv = None

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Visualization defaults
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def download_heart_disease_data(output_dir='data/raw'):
    """Download Heart Disease dataset from UCI ML Repository and save CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    if fetch_ucirepo is None:
        raise RuntimeError("ucimlrepo is not installed. Install or run without download.")

    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    full_data = pd.concat([X, y], axis=1)

    output_path = os.path.join(output_dir, 'heart_disease_raw.csv')
    full_data.to_csv(output_path, index=False)
    X.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
    y.to_csv(os.path.join(output_dir, 'targets.csv'), index=False)

    return output_path


def load_raw_data(data_path='data/raw/heart_disease_raw.csv'):
    df = pd.read_csv(data_path)
    return df


def handle_missing_values(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    target_cols = ['num', 'target', 'condition']
    for target_col in target_cols:
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

    if len(numerical_cols) > 0 and df[numerical_cols].isnull().sum().sum() > 0:
        num_imputer = SimpleImputer(strategy='median')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    if len(categorical_cols) > 0 and df[categorical_cols].isnull().sum().sum() > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df


def encode_features(df):
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    target_cols = ['num', 'target', 'condition']
    for target_col in target_cols:
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

    if len(categorical_cols) == 0:
        return df

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def prepare_target_variable(df):
    possible_targets = ['num', 'target', 'condition']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]

    df['target_binary'] = (df[target_col] > 0).astype(int)
    return df


def save_processed_data(df, output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'heart_disease_processed.csv')
    df.to_csv(output_path, index=False)
    return output_path


def preprocess_pipeline(input_path='data/raw/heart_disease_raw.csv', output_dir='data/processed'):
    df = load_raw_data(input_path)
    df = handle_missing_values(df)
    df = encode_features(df)
    df = prepare_target_variable(df)
    out = save_processed_data(df, output_dir)
    return df, out


def load_processed_data(data_path='data/processed/heart_disease_processed.csv'):
    return pd.read_csv(data_path)


def create_output_directory(output_dir='outputs/eda_visualizations'):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_class_balance(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    target_counts = df['target_binary'].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']
    axes[0].bar(['No Disease (0)', 'Disease (1)'], target_counts.values, color=colors, alpha=0.8, edgecolor='black')
    for i, v in enumerate(target_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', va='bottom')
    axes[1].pie(target_counts.values, labels=['No Disease (0)', 'Disease (1)'], autopct='%1.1f%%', colors=colors, startangle=90)
    plt.tight_layout()
    output_path = os.path.join(output_dir, '01_class_balance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def plot_numerical_distributions(df, output_dir):
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['target_binary', 'num', 'target', 'condition']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    n_cols = 4
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten() if len(numerical_cols) > 1 else [axes]
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        ax.hist(df[col].dropna(), bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        mean_val = df[col].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2)
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, '02_feature_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def plot_correlation_heatmap(df, output_dir):
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, '03_correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def plot_feature_vs_target(df, output_dir):
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['target_binary', 'num', 'target', 'condition']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    if 'target_binary' in df.columns:
        corr_with_target = df[numerical_cols].corrwith(df['target_binary']).abs()
        top_features = corr_with_target.nlargest(8).index.tolist()
    else:
        top_features = numerical_cols[:8]
    n_cols = 4
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten() if len(top_features) > 1 else [axes]
    for idx, col in enumerate(top_features):
        ax = axes[idx]
        df.boxplot(column=col, by='target_binary', ax=ax, patch_artist=True)
        plt.sca(ax)
        plt.xticks([1, 2], ['0', '1'])
    for idx in range(len(top_features), len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, '04_features_by_target.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def generate_sweetviz_report(df, output_dir):
    if sv is None:
        return None
    report = sv.analyze(df, target_feat='target_binary' if 'target_binary' in df.columns else None)
    output_path = os.path.join(output_dir, 'sweetviz_eda_report.html')
    report.show_html(output_path, open_browser=False)
    return output_path


def generate_summary_statistics(df, output_dir):
    summary_stats = df.describe()
    output_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_stats.to_csv(output_path)
    return output_path


def run_eda_pipeline(data_path='data/processed/heart_disease_processed.csv', output_dir='outputs/eda_visualizations'):
    df = load_processed_data(data_path)
    output_dir = create_output_directory(output_dir)
    plot_class_balance(df, output_dir)
    plot_numerical_distributions(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_feature_vs_target(df, output_dir)
    generate_summary_statistics(df, output_dir)
    generate_sweetviz_report(df, output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Combined data pipeline (download, preprocess, eda)')
    parser.add_argument('--download', action='store_true', help='Download raw data from UCI')
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing pipeline')
    parser.add_argument('--eda', action='store_true', help='Run EDA on processed data')
    parser.add_argument('--all', action='store_true', help='Run all steps: download -> preprocess -> eda')
    args = parser.parse_args()

    # If no flags provided, run everything
    if not any([args.download, args.preprocess, args.eda, args.all]):
        args.all = True

    if args.download or args.all:
        # Download only if ucimlrepo is available; otherwise skip and continue
        if fetch_ucirepo is None:
            print('ucimlrepo not installed; skipping download and continuing')
        else:
            print('Downloading raw data...')
            download_heart_disease_data()

    if args.preprocess or args.all:
        print('Running preprocessing...')
        df, out = preprocess_pipeline()
        print(f'Processed data saved to: {out}')

    if args.eda or args.all:
        print('Running EDA...')
        out_dir = run_eda_pipeline()
        print(f'EDA outputs saved to: {out_dir}')


if __name__ == '__main__':
    main()
