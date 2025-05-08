# Enhancing Road Safety with AI-Driven Traffic Accident Analysis and Prediction
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
https://github.com/Rakesh-ai-ds/AI-driven-traffic-accident-analysis-and-prediction/blob/9ceab8afdaaba550790a395fbd3758fbc0f19a01/WhatsApp%20Image%202025-05-09%20at%2000.23.23_609a3c82.jpg

## Project Overview
This project leverages machine learning to analyze traffic accident data and predict accident severity, ultimately contributing to enhanced road safety measures. By identifying patterns and risk factors in historical accident data, the system helps recognize conditions that lead to more severe accidents.

## Features
- **Data Processing**: Cleans and preprocesses US accident data to prepare for analysis
- **Exploratory Data Analysis**: Visualizes key patterns in accident data
- **Severity Prediction**: Implements a Random Forest model to predict accident severity
- **Feature Importance Analysis**: Identifies the most influential factors affecting accident severity
- **Comprehensive Visualization**: Generates multiple plots to illustrate findings

## Dataset
The project uses the "US_Accidents_March23.csv" dataset, which contains information about traffic accidents across the United States. Key features include:
- Geographic coordinates (latitude/longitude)
- Distance
- Weather conditions (temperature, humidity, visibility, wind speed, precipitation)
- Accident severity (target variable)

## Installation

### Prerequisites
- Python 3.10
- pip package manager
- VS Code (recommended IDE)

### Environment Setup
1. Clone this repository:
```bash
git clone https://github.com/yourusername/road-safety-prediction.git
cd road-safety-prediction
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset "US_Accidents_March23.csv" and place it in the project root directory.

### VS Code Configuration
This project includes a `.vscode/launch.json` file for debugging in Visual Studio Code:

1. Open the project in VS Code:
```bash
code .
```

2. Install recommended VS Code extensions:
   - Python extension by Microsoft
   - Pylance for improved language support
   - Jupyter for notebook support (optional)

3. Select the Python interpreter from your virtual environment:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from your virtual environment

### Troubleshooting Common Errors

#### ImportError or ModuleNotFoundError
If you encounter errors like `ImportError: No module named X`:
```bash
# Make sure your virtual environment is activated, then try:
pip install X
# Or update all dependencies:
pip install --upgrade -r requirements.txt
```

#### Dataset Loading Issues
If you encounter errors loading the dataset:
- Verify the CSV file is in the correct location
- Check file permissions
- Try with a smaller sample first:
```python
df = pd.read_csv('US_Accidents_March23.csv', nrows=1000)
```

#### Memory Errors
For "MemoryError" when loading large datasets:
```python
# Try loading with chunks
chunks = pd.read_csv('US_Accidents_March23.csv', chunksize=10000)
df = pd.concat([chunk for chunk in chunks], ignore_index=True)
```

#### CUDA/GPU Errors
If using GPU acceleration and encountering errors:
- Ensure compatible CUDA drivers are installed
- Fall back to CPU processing by modifying the code to avoid GPU-specific operations

## Usage
Run the main script to perform the complete analysis and model training:
```bash
python main.py
```

The script will:
1. Load and sample the dataset
2. Preprocess the data
3. Train a Random Forest classifier
4. Evaluate model performance
5. Generate visualizations in the "plots" directory

## Generated Visualizations
The project automatically generates the following visualizations:
1. **Severity Distribution**: Shows the distribution of accident severity levels
2. **Correlation Heatmap**: Displays correlations between numerical features
3. **Weather Impact**: Illustrates how different weather conditions affect accident severity
4. **Feature Importance**: Ranks features by their importance in predicting severity
5. **Confusion Matrix**: Evaluates model prediction accuracy

## Model Performance Evaluation

### Metrics Explained
The RandomForest classifier produces the following metrics for accident severity prediction:

- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. Higher precision means fewer false positive errors.
- **Recall**: The ratio of correctly predicted positive observations to all actual positives. Higher recall means fewer false negative errors.
- **F1-score**: The weighted average of Precision and Recall. This score considers both false positives and false negatives.
- **Support**: The number of actual occurrences of the class in the dataset.

### Interpreting Results
A sample classification report might look like:
```
              precision    recall  f1-score   support
           1       0.85      0.79      0.82       450
           2       0.72      0.81      0.76       650
           3       0.91      0.87      0.89       350
           4       0.94      0.92      0.93       150

    accuracy                          0.83      1600
```

### Improving Model Accuracy
If model accuracy needs improvement:

1. Try different algorithms:
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Replace RandomForestClassifier with:
GradientBoostingClassifier(random_state=42)
# or
SVC(kernel='rbf', probability=True, random_state=42)
```

2. Implement cross-validation for more robust evaluation:
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Average accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
```
If the model training is slow or resource-intensive:

1. Reduce sample size for initial testing:
```python
# In main.py, modify the sample_size parameter
df = load_sample_data(file_path, sample_size=5000)  # Try with smaller sample
```

2. Optimize RandomForest parameters:
```python
# Modify the classifier in the pipeline
RandomForestClassifier(
    n_estimators=50,  # Reduce from default 100
    max_depth=10,     # Limit tree depth
    min_samples_split=5,
    random_state=42,
    n_jobs=-1  # Use all available processors
)
```

3. Selective feature processing:
```python
# Use only the most important features
features = ['distance(mi)', 'temperature(f)', 'humidity(%)', 'weather_condition']
```

### Environment-Specific Configurations

#### Windows-Specific Setup
On Windows systems, you might encounter path-related issues:
```python
# Replace this:
os.makedirs('plots', exist_ok=True)

# With this:
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(plots_dir, exist_ok=True)
```

#### macOS/Linux Setup
On Unix-based systems, ensure proper permissions:
```bash
# Make scripts executable
chmod +x main.py

# If using a launcher script
chmod +x run.sh
```

#### Docker Support
For containerized deployment, a basic Dockerfile is provided:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

To build and run:
```bash
docker build -t road-safety-prediction .
docker run road-safety-prediction
```

## Project Structure
```
road-safety-prediction/
├── main.py              # Main script for data processing and model training
├── README.md            # Project documentation
├── requirements.txt     # Required dependencies
├── US_Accidents_March23.csv  # Dataset (not included in repository)
├── plots/               # Generated visualizations
│   ├── severity_distribution.png
│   ├── correlation_heatmap.png
│   ├── weather_severity.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
└── .vscode/             # VS Code configuration
    └── launch.json      # Debug configuration
```

## Future Improvements
- Implement hyperparameter tuning to optimize model performance
- Add geospatial analysis to identify high-risk accident locations
- Develop interactive dashboards for real-time risk assessment
- Expand feature engineering to incorporate road conditions and traffic density
- Deploy model as a web service for real-time predictions

## License
This project is licensed under the MIT License - see the LICENSE file for details. This is a fully open-source project with no restrictions on use, modification, or distribution.

### Open Source Commitment
This project is committed to the principles of open-source software:
- **Free to use**: Anyone can use this code without payment
- **Free to modify**: The code can be adapted to suit any need
- **Free to distribute**: The software can be shared with anyone
- **Transparent**: All source code is publicly available
- **Community-driven**: Contributions and improvements are welcome from all

## Acknowledgments
- The dataset used in this project is derived from public sources
- Special thanks to contributors of scikit-learn, pandas, matplotlib, and seaborn libraries
