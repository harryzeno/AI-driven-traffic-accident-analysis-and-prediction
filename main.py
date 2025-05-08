import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load 10,000-row sample from dataset
def load_sample_data(file_path, sample_size=10000):
    logging.info(f"Loading {sample_size} samples from {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
        if len(df) < sample_size:
            logging.warning(f"Dataset has only {len(df)} rows, using all.")
            return df
        df = df.sample(n=sample_size, random_state=42)
        logging.info(f"Loaded {len(df)} rows.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise

# Preprocess data for road safety prediction
def preprocess_data(df):
    logging.info("Preprocessing data...")
    
    # Lowercase column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Select features relevant to road safety
    features = ['start_lat', 'start_lng', 'distance(mi)', 'temperature(f)', 
                'humidity(%)', 'visibility(mi)', 'wind_speed(mph)', 
                'precipitation(in)', 'weather_condition']
    target = 'severity'
    
    # Check if required columns exist
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Drop rows with missing severity
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)
    
    # Define features and target
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define numerical and categorical columns
    numerical_cols = ['start_lat', 'start_lng', 'distance(mi)', 'temperature(f)', 
                     'humidity(%)', 'visibility(mi)', 'wind_speed(mph)', 
                     'precipitation(in)']
    categorical_cols = ['weather_condition']
    
    # Preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

# Train and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    logging.info("Training RandomForest model for accident severity prediction...")
    
    # Create pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    print("\nClassification Report for Accident Severity Prediction:")
    print(classification_report(y_test, y_pred, zero_division=0))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Define global column lists
NUMERICAL_COLS = ['start_lat', 'start_lng', 'distance(mi)', 'temperature(f)', 
                  'humidity(%)', 'visibility(mi)', 'wind_speed(mph)', 'precipitation(in)']
CATEGORICAL_COLS = ['weather_condition']

# Create all visualizations
def create_visualizations(df, model, preprocessor, X_test, y_test):
    os.makedirs('plots', exist_ok=True)
    
    # 1. Distribution of Accident Severity
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='severity')
    plt.title('Accident Severity Distribution')
    plt.xlabel('Severity')
    plt.ylabel('Count')
    plt.savefig('plots/severity_distribution.png')
    plt.close()
    
    # 2. Correlation Heatmap of Numerical Features
    plt.figure(figsize=(10, 8))
    corr_matrix = df[NUMERICAL_COLS].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Numerical Features Correlation')
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    
    # 3. Weather Condition Impact on Severity
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='weather_condition', hue='severity')
    plt.title('Weather Impact on Severity')
    plt.xlabel('Weather Condition')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Severity')
    plt.tight_layout()
    plt.savefig('plots/weather_severity.png')
    plt.close()
    
    # 4. Feature Importance Plot
    feature_importance = model.named_steps['classifier'].feature_importances_
    feature_names = (NUMERICAL_COLS + 
        model.named_steps['preprocessor']
            .named_transformers_['cat']
            .named_steps['onehot']
            .get_feature_names_out(CATEGORICAL_COLS).tolist())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_names)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    # 5. Confusion Matrix
    y_pred = model.predict(X_test)
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Accident Severity')
    plt.xlabel('Predicted Severity')
    plt.ylabel('Actual Severity')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    return model

# Main execution
if __name__ == "__main__":
    file_path = 'US_Accidents_March23.csv'
    print("Project: Enhancing Road Safety with AI-Driven Traffic Accident Analysis and Prediction")
    
    # Load data
    df = load_sample_data(file_path)
    
    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Train and evaluate
    model = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    
    logging.info("Project completed successfully.")