import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
fake = Faker()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# -------------------------------
# Risk score calculation
# -------------------------------
def calculate_risk(row):
    risk = 0
    risk += (row['Age'] / 85) * 20  # age
    if row['Cholesterol'] > 240: 
        risk += 20
    elif row['Cholesterol'] > 200: 
        risk += 10
    if row['Blood_Sugar'] > 126: 
        risk += 15
    elif row['Blood_Sugar'] > 100: 
        risk += 8
    if row['Smoking'] == 'Yes': 
        risk += 15
    if row['Family_History_Cancer'] == 'Yes': 
        risk += 20
    risk += min(row['Tumor_Size_cm'] * 3, 15)  # tumor size
    return min(risk, 100)

# -------------------------------
# Add anomalies
# -------------------------------
def add_anomalies(data, fraction_nulls=0.02, fraction_duplicates=0.02,
                  fraction_outliers=0.02, fraction_incorrect=0.02,
                  random_state=42):
    np.random.seed(random_state)
    n_rows = len(data)
    numeric_cols = ['Cholesterol', 'Blood_Sugar', 'BMI', 'Tumor_Size_cm']

    # Nulls
    null_indices = np.random.choice(data.index, int(n_rows * fraction_nulls), replace=False)
    for idx in null_indices:
        col = np.random.choice(data.columns)
        data.loc[idx, col] = np.nan

    # Duplicates
    duplicate_indices = np.random.choice(data.index, int(n_rows * fraction_duplicates), replace=False)
    duplicates = data.loc[duplicate_indices].copy()
    data = pd.concat([data, duplicates], ignore_index=True)

    # Outliers (multiply some values by big factor)
    outlier_indices = np.random.choice(data.index, int(n_rows * fraction_outliers), replace=False)
    for col in numeric_cols:
        data.loc[outlier_indices, col] *= np.random.uniform(3, 10, size=len(outlier_indices))

    # Incorrect formats(Some numeric fields (Cholesterol, Blood_Sugar) are replaced with "abc" → simulates dirty text in numbers.)
    incorrect_indices = np.random.choice(data.index, int(n_rows * fraction_incorrect), replace=False)
    for col in ['Cholesterol', 'Blood_Sugar']:
        data[col] = data[col].astype(object)
        data.loc[incorrect_indices, col] = "abc"

    return data

# -------------------------------
# Generate dataset
# -------------------------------
def generate_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    data = pd.DataFrame({
        'Patient_ID': range(1, n_samples + 1),
        'Name': [fake.name() for _ in range(n_samples)],
        'Age': np.random.randint(18, 85, size=n_samples),
        'Gender': np.random.choice(['Male', 'Female'], size=n_samples, p=[0.48, 0.52]),
        'Height_cm': np.random.normal(170, 10, size=n_samples),
        'Weight_kg': np.random.normal(75, 15, size=n_samples),
        'BMI': np.random.uniform(18, 35, size=n_samples),
        'Systolic_BP': np.random.normal(120, 20, size=n_samples),
        'Diastolic_BP': np.random.normal(80, 15, size=n_samples),
        'Heart_Rate': np.random.normal(70, 12, size=n_samples),
        'Temperature_F': np.random.normal(98.6, 1.5, size=n_samples),
        'Blood_Sugar': np.random.normal(100, 30, size=n_samples),
        'Cholesterol': np.random.normal(200, 40, size=n_samples),
        'Hemoglobin': np.random.normal(14, 2, size=n_samples),
        'Smoking': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.25, 0.75]),
        'Alcohol_Consumption': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7]),
        'Exercise_Hours_Week': np.random.poisson(3, size=n_samples),
        'Diabetes': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.15, 0.85]),
        'Hypertension': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.20, 0.80]),
        'Heart_Disease': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.10, 0.90]),
        'Hospital_Visits_Year': np.random.poisson(2, size=n_samples),
        'Insurance_Type': np.random.choice(['Public', 'Private', 'Uninsured'], size=n_samples, p=[0.4, 0.5, 0.1]),
        'Family_History_Cancer': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.18, 0.82]),
        'Biopsy_Performed': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.06, 0.94]),
        'Biopsy_Result': np.random.choice(['Benign', 'Malignant', 'Unknown'], size=n_samples, p=[0.85, 0.10, 0.05]),
        'Treatment': np.random.choice(['Chemotherapy', 'Radiation Therapy', 'Surgery', 'Immunotherapy', 'None'],
                                      size=n_samples, p=[0.2, 0.15, 0.25, 0.1, 0.3]),
        'Allergies': np.random.choice(['None', 'Penicillin', 'Peanuts', 'Seafood', 'Dust', 'Pollen'],
                                      size=n_samples, p=[0.5, 0.15, 0.1, 0.1, 0.08, 0.07]),
        'Sonography_Result': np.random.choice(['Normal', 'Abnormal'], size=n_samples, p=[0.85, 0.15]),
        'Tumor_Size_cm': np.round(np.random.uniform(0, 10, size=n_samples), 2)
    })

    data['BMI'] = (data['Weight_kg'] / ((data['Height_cm'] / 100) ** 2))
    data['Risk_Score'] = data.apply(calculate_risk, axis=1)
    return add_anomalies(data, random_state=random_state)

# -------------------------------
# Cleaning with IQR + Z-Score + Rules
# -------------------------------
def clean_dataset(df, z_thresh=3):
    print("Initial shape:", df.shape)

    # 1. Handle Missing(nulls)
    df = df.dropna()

    # 2. Remove Duplicates
    df = df.drop_duplicates()

    # 3. Fix incorrect data types
    for col in ['Cholesterol', 'Blood_Sugar']:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=['Cholesterol', 'Blood_Sugar'])

    # 4. Outlier detection using IQR
    numeric_cols = ['Cholesterol', 'Blood_Sugar', 'BMI', 'Tumor_Size_cm']
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # 5. Outlier detection using Z-score
    for col in numeric_cols:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        df = df[np.abs(z_scores) < z_thresh]

    # 6. Domain rule filtering
    df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]
    df = df[(df['BMI'] >= 10) & (df['BMI'] <= 60)]
    df = df[(df['Blood_Sugar'] >= 40) & (df['Blood_Sugar'] <= 500)]
    df = df[(df['Cholesterol'] >= 80) & (df['Cholesterol'] <= 400)]

    print("Cleaned shape:", df.shape)
    return df

# -------------------------------
# Run full pipeline
# -------------------------------
# -------------------------------
if __name__ == "__main__":
    TARGET_ROWS = 1000

    print("Generating dataset with anomalies...")
    dataset = generate_data(TARGET_ROWS)
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    dataset[numeric_cols] = dataset[numeric_cols]
    dataset.to_csv("patient_data_with_anomalies.csv", index=False)
    dataset.to_excel("patient_data_with_anomalies.xlsx", index=False)  # <-- This is the Excel export

    print("Cleaning dataset (IQR + Z-Score + Rules)...")
    cleaned = clean_dataset(dataset)

    # If rows are missing after cleaning
    if len(cleaned) < TARGET_ROWS:
        extra_needed = TARGET_ROWS - len(cleaned)
        print(f"⚠️ Only {len(cleaned)} rows after cleaning. Generating {extra_needed*2} more...")

        # Generate a bigger batch of extra data
        extra_data = generate_data(extra_needed * 2, random_state=np.random.randint(0, 99999))
        extra_cleaned = clean_dataset(extra_data)

        # Merge original + extra
        cleaned = pd.concat([cleaned, extra_cleaned], ignore_index=True)

    #  Ensure exactly 1000 rows (cut off extras if too many)
    cleaned = cleaned.head(TARGET_ROWS)

    #  ROUND numeric columns to 2 decimal places
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    cleaned[numeric_cols] = cleaned[numeric_cols]

    print(f"✅ Final dataset has exactly {len(cleaned)} rows")

    #  SAVE to CSV and Excel
    cleaned.to_csv("patient_data_cleaned.csv", index=False)
    cleaned.to_excel("patient_data_cleaned.xlsx", index=False)  # <-- This is the Excel export

    print("Saved cleaned dataset to CSV and Excel.")

# -------------------------------
# Data Integrity Checker
# -------------------------------
def check_data_integrity(df):
    issues = []

    # Convert to numeric safely
    for col in ['Cholesterol', 'Blood_Sugar']:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # turns "abc" into NaN

    # 1. Check data types
    expected_types = {
        'Age': (int, float),
        'Cholesterol': (int, float),
        'Blood_Sugar': (int, float),
        'BMI': (int, float),
        'Tumor_Size_cm': (int, float),
        'Smoking': str,
        'Family_History_Cancer': str
    }

    for col, typ in expected_types.items():
        if not df[col].map(lambda x: isinstance(x, typ)).all():
            issues.append(f"Invalid data types in column: {col}")

    # 2. Check value ranges
    if df['Age'].dropna().lt(0).any() or df['Age'].dropna().gt(120).any():
        issues.append("Age out of expected range (0-120)")
    if df['BMI'].dropna().lt(10).any() or df['BMI'].dropna().gt(60).any():
        issues.append("BMI out of expected range (10-60)")
    if df['Blood_Sugar'].dropna().lt(40).any() or df['Blood_Sugar'].dropna().gt(500).any():
        issues.append("Blood Sugar out of expected range (40-500)")
    if df['Cholesterol'].dropna().lt(80).any() or df['Cholesterol'].dropna().gt(400).any():
        issues.append("Cholesterol out of expected range (80-400)")
    if df['Tumor_Size_cm'].dropna().lt(0).any() or df['Tumor_Size_cm'].dropna().gt(20).any():
        issues.append("Tumor size out of range (0-20)")

    # 3. Check categorical values
    for col, allowed in {
        'Smoking': ['Yes', 'No'],
        'Family_History_Cancer': ['Yes', 'No'],
        'Gender': ['Male', 'Female']
    }.items():
        if not df[col].isin(allowed).all():
            issues.append(f"Unexpected values in column: {col}")

    # 4. Logical consistency
    if 'Systolic_BP' in df.columns and 'Diastolic_BP' in df.columns:
        inconsistent_bp = (df['Systolic_BP'] < df['Diastolic_BP']).sum()
        if inconsistent_bp > 0:
            issues.append(f"{inconsistent_bp} rows where Diastolic BP > Systolic BP")

    # Output results
    if issues:
        print("❌ Data Integrity Issues Found:")
        with open("data_integrity_issues.txt", "w") as f:
            for i in issues:
                print(f" - {i}")
                f.write(f"{i}\n")
        print("📄 Saved issues to data_integrity_issues.txt")

    else:
        print("✅ Data Integrity Check Passed")


# -------------------------------
# Run full pipeline with integrity checks
# -------------------------------
if __name__ == "__main__":
    print("Generating dataset with anomalies...")
    dataset = generate_data(1000)

    # print("\n🔎 Checking integrity BEFORE cleaning...")
    # check_data_integrity(dataset)

    # Save dataset with anomalies
    # Save dataset with anomalies
    dataset.to_csv("patient_data_with_anomalies.csv", index=False)
    print(" Saved: patient_data_with_anomalies.csv")

# Save cleaned dataset
    cleaned.to_csv("patient_data_cleaned.csv", index=False)
    print(" Saved: patient_data_cleaned.csv")

# Save integrity-checked dataset



    # Final integrity check - save only if passed
    print("\n🧪 Saving only integrity-passed data...")
    def passes_integrity(df):
        issues = []
        # Convert columns safely
        for col in ['Cholesterol', 'Blood_Sugar']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for NaNs
        if df[['Cholesterol', 'Blood_Sugar']].isna().any().any():
            return False
        if df['Age'].between(0, 120).all() and \
           df['BMI'].between(10, 60).all() and \
           df['Blood_Sugar'].between(40, 500).all() and \
           df['Cholesterol'].between(80, 400).all():
            return True
        return False

    if passes_integrity(cleaned):
        cleaned.to_csv("patient_data_integrity_checked.csv", index=False)
        print(" Saved: patient_data_integrity_checked.csv")
    else:
        print("Integrity check failed. File not saved.")



# pd.get_dummies()

def prepare_features_and_target(data):

    #feature is engineering is used to modify the feature in dataset to improve performance of machine learning 
    processed_data = data.copy()

    # Create target variable based on risk assessment
    risk_threshold = processed_data['Risk_Score'].median() #calculate median of risk_score
    processed_data['High_Risk'] = (processed_data['Risk_Score'] > risk_threshold).astype(int)   #if risk_score is > threshold then it considerd as high risk factor

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Gender', 'Smoking', 'Alcohol_Consumption', 'Diabetes',
                          'Hypertension', 'Heart_Disease', 'Insurance_Type']
    categorical_columns += ['Treatment', 'Biopsy_Result', 'Sonography_Result']
    processed_data = pd.get_dummies(processed_data, columns=categorical_columns, drop_first=True)

    for col in categorical_columns:
        if col in processed_data.columns:
            le = LabelEncoder() #turns categorical values (strings) into integer labels
            processed_data[col + '_encoded'] = le.fit_transform(processed_data[col])
            label_encoders[col] = le

    # Select feature columns
    #numeric column used to predict the cancer
    feature_columns = [
        'Age', 'Height_cm', 'Weight_kg', 'BMI', 'Systolic_BP', 'Diastolic_BP',
        'Heart_Rate', 'Temperature_F', 'Blood_Sugar', 'Cholesterol', 'Hemoglobin',
        'Exercise_Hours_Week', 'Hospital_Visits_Year', 'Gender_encoded',
        'Smoking_encoded', 'Alcohol_Consumption_encoded', 'Diabetes_encoded',
        'Hypertension_encoded', 'Heart_Disease_encoded', 'Insurance_Type_encoded',
        'Tumor_Size_cm'  # ✅ keep these
    ]

    # Filter existing columns
    available_features = [col for col in feature_columns if col in processed_data.columns]

    X = processed_data[available_features]
    y = processed_data['High_Risk']

    return X, y, label_encoders, processed_data

def save_feature_info(X, y, label_encoders):
    feature_info = {
        'feature_names': list(X.columns),
        'target_name': 'High_Risk',
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'target_distribution': y.value_counts().to_dict()
    }

    # Save encoders and feature info
    import pickle
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    with open('feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)

    return feature_info

if __name__ == "__main__":
    print("Preparing features and target variable...")
    data = pd.read_csv('patient_data_integrity_checked.csv')

    X, y, label_encoders, processed_data = prepare_features_and_target(data)
    feature_info = save_feature_info(X, y, label_encoders)

    # Save features and target
    X.to_csv('features.csv', index=False)
    y.to_csv('target.csv', index=False)
    processed_data.to_csv('processed_data.csv', index=False)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {feature_info['target_distribution']}")
    print(f"Feature columns: {len(feature_info['feature_names'])}")
    print("Saved files: features.csv, target.csv, processed_data.csv")
    print("Saved encoders: label_encoders.pkl, feature_info.pkl")

def scale_and_split_data(X, y, test_size=0.2, random_state=42):
    # Train-test split
    #splits the dataset into training and testing sets and then scales the features using standardization (z-score normalization)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()   #standerdizes the feature
    X_train_scaled = scaler.fit_transform(X_train)  #Computes mean & std on training data
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_split_data(X_train, X_test, y_train, y_test, scaler):
    # Save datasets
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    split_info = {
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'train_ratio': X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]),
        'test_ratio': X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]),
        'train_target_dist': y_train.value_counts().to_dict(),
        'test_target_dist': y_test.value_counts().to_dict()
    }

    with open('split_info.pkl', 'wb') as f:
        pickle.dump(split_info, f)

    return split_info

if __name__ == "__main__":
    print("Loading features and target, then scaling and splitting...")
    X = pd.read_csv('features.csv')
    y = pd.read_csv('target.csv').squeeze()

    X_train, X_test, y_train, y_test, scaler = scale_and_split_data(X, y)
    split_info = save_split_data(X_train, X_test, y_train, y_test, scaler)

    print(f"Training set: {split_info['train_size']} samples ({split_info['train_ratio']:.2%})")
    print(f"Test set: {split_info['test_size']} samples ({split_info['test_ratio']:.2%})")
    print(f"Train target distribution: {split_info['train_target_dist']}")
    print(f"Test target distribution: {split_info['test_target_dist']}")
    print("Saved files: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print("Saved objects: scaler.pkl, split_info.pkl")

#-------- ML MODEL AND EVALUATION PART ----------#
def train_multiple_models(X_train, y_train):
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    trained_models = {}
    model_scores = {}

    for name, model in models.items():
        print(f"Training {name}...")

        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        model_scores[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }

        print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return trained_models, model_scores

def select_best_model(model_scores):
    best_model_name = max(model_scores, key=lambda x: model_scores[x]['cv_mean'])
    best_score = model_scores[best_model_name]['cv_mean']

    return best_model_name, best_score

def save_models(trained_models, model_scores, best_model_name):
    # Save all trained models
    with open('trained_models.pkl', 'wb') as f:
        pickle.dump(trained_models, f)

    # Save model scores
    with open('model_scores.pkl', 'wb') as f:
        pickle.dump(model_scores, f)

    # Save best model separately
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(trained_models[best_model_name], f)

    # Save model info
    model_info = {
        'best_model_name': best_model_name,
        'best_model_score': model_scores[best_model_name]['cv_mean'],
        'all_model_scores': {name: scores['cv_mean'] for name, scores in model_scores.items()}
    }

    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)

    return model_info

if __name__ == "__main__":
    print("Loading training data and training models...")
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()

    trained_models, model_scores = train_multiple_models(X_train, y_train)
    best_model_name, best_score = select_best_model(model_scores)
    model_info = save_models(trained_models, model_scores, best_model_name)

    print(f"\nModel Selection Results:")
    print(f"Best model: {best_model_name}")
    print(f"Best CV score: {best_score:.4f}")
    print(f"\nAll model scores:")
    for name, score in model_info['all_model_scores'].items():
        print(f"  {name}: {score:.4f}")

    print("\nSaved files: trained_models.pkl, model_scores.pkl, best_model.pkl, model_info.pkl")


#-------- VISUALIZATION PART ---------#
def evaluate_all_models(trained_models, X_test, y_test):
    results = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

    return results

def create_visualizations(results, y_test):
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))

    # 1. Model Comparison Bar Plot
    plt.subplot(2, 3, 1)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(results.keys())

    x = np.arange(len(model_names))
    width = 0.2

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        plt.bar(x + i*width, values, width, label=metric.capitalize())

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*1.5, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. ROC Curves
    plt.subplot(2, 3, 2)
    for name, result in results.items():
        if result['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            auc_score = result['auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Best Model Confusion Matrix
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    plt.subplot(2, 3, 3)
    cm = results[best_model_name]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # 4. Feature Importance (if available)
    if hasattr(list(results.values())[0], 'feature_importances_'):
        plt.subplot(2, 3, 4)
        # Get feature names
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)

        model = None
        for name, trained_model in results.items():
            if hasattr(trained_model, 'feature_importances_'):
                model = trained_model
                break

        if model is not None:
            feature_importance = pd.DataFrame({
                'feature': feature_info['feature_names'],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)

            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance')

    # 5. Accuracy Comparison
    plt.subplot(2, 3, 5)
    accuracies = [results[model]['accuracy'] for model in model_names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = plt.bar(model_names, accuracies, color=colors)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')

    # 6. AUC Comparison
    plt.subplot(2, 3, 6)
    auc_scores = [results[model]['auc'] for model in model_names if results[model]['auc'] is not None]
    auc_models = [model for model in model_names if results[model]['auc'] is not None]

    if auc_scores:
        colors = plt.cm.plasma(np.linspace(0, 1, len(auc_models)))
        bars = plt.bar(auc_models, auc_scores, color=colors)
        plt.xlabel('Models')
        plt.ylabel('AUC Score')
        plt.title('Model AUC Comparison')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, auc in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_evaluation_results(results):
    # Create summary DataFrame
    summary_data = []
    for model, metrics in results.items():
        summary_data.append({
            'Model': model,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1'],
            'AUC': metrics['auc'] if metrics['auc'] is not None else 'N/A'
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('model_evaluation_summary.csv', index=False)

    # Save detailed results
    with open('evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return summary_df


if __name__ == "__main__":
    print("Loading models and test data for evaluation...")

    # Load trained models and test data
    with open('trained_models.pkl', 'rb') as f:
        trained_models = pickle.load(f)

    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').squeeze()

    # Evaluate all models
    results = evaluate_all_models(trained_models, X_test, y_test)

    # Create visualizations
    create_visualizations(results, y_test)

    # Save results
    summary_df = save_evaluation_results(results)

    print("\nModel Evaluation Summary:")
    print(summary_df.to_string(index=False))

    best_model = summary_df.loc[summary_df['Accuracy'].idxmax(), 'Model']
    best_accuracy = summary_df['Accuracy'].max()

    print(f"\nBest performing model: {best_model}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    # Save best model info
    model_info = {
        "best_model_name": best_model,
        "best_model_score": best_accuracy
    }
    with open("model_info.pkl", "wb") as f:
        pickle.dump(model_info, f)

    print(f"📄 Saved model_info.pkl with {best_model_name}")

    print("\nSaved files: model_evaluation_summary.csv, evaluation_results.pkl, model_evaluation_results.png")


    # -------PREDICTION -------#
class ModelInference:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_info = None
        self.model_info = None

    def load_model_components(self):
        """Load all necessary components for inference"""
        try:
            # Load best model
            with open('best_model.pkl', 'rb') as f:
                self.model = pickle.load(f)

            # Load scaler
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)

            # Load label encoders
            with open('label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)

            # Load feature info
            with open('feature_info.pkl', 'rb') as f:
                self.feature_info = pickle.load(f)

            # Load model info
            with open('model_info.pkl', 'rb') as f:
                self.model_info = pickle.load(f)

            print(f"Loaded {self.model_info['best_model_name']} model successfully")
            print(f"Model CV accuracy: {self.model_info['best_model_score']:.4f}")
            print(f"⚡ Actual loaded model type: {type(self.model).__name__}")


        except FileNotFoundError as e:
            print(f"Error loading model components: {e}")
            print("Make sure all model files exist in the current directory")

    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        processed_data = input_data.copy()

        # Encode categorical variables
        categorical_mappings = {
            'Gender': ['Male', 'Female'],
            'Smoking': ['Yes', 'No'],
            'Alcohol_Consumption': ['Yes', 'No'],
            'Diabetes': ['Yes', 'No'],
            'Hypertension': ['Yes', 'No'],
            'Heart_Disease': ['Yes', 'No'],
            'Insurance_Type': ['Public', 'Private', 'Uninsured']
        }

        for col, encoder in self.label_encoders.items():
            if col in processed_data.columns:
                try:
                    processed_data[col + '_encoded'] = encoder.transform(processed_data[col])
                except ValueError:
                    # Handle unseen categories
                    print(f"Warning: Unknown category in {col}, using most frequent class")
                    processed_data[col + '_encoded'] = 0

        # Select only the features used in training
        feature_columns = self.feature_info['feature_names']
        available_features = [col for col in feature_columns if col in processed_data.columns]

        if len(available_features) != len(feature_columns):
            missing_features = set(feature_columns) - set(available_features)
            print(f"Warning: Missing features: {missing_features}")

        X = processed_data[available_features]

        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=available_features)

        return X_scaled

    def predict_single(self, patient_data):
        """Make prediction for a single patient"""
        if self.model is None:
            self.load_model_components()

        # Convert to DataFrame if it's a dictionary
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])

        # Preprocess
        X_processed = self.preprocess_input(patient_data)

        # Make prediction
        prediction = self.model.predict(X_processed)[0]
        probability = self.model.predict_proba(X_processed)[0] if hasattr(self.model, 'predict_proba') else None

        result = {
            'prediction': int(prediction),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability_low_risk': float(probability[0]) if probability is not None else None,
            'probability_high_risk': float(probability[1]) if probability is not None else None
        }

        return result

    def predict_batch(self, data_file):
        """Make predictions for a batch of patients"""
        if self.model is None:
            self.load_model_components()

        # Load data
        data = pd.read_csv(data_file)

        # Preprocess
        X_processed = self.preprocess_input(data)

        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed) if hasattr(self.model, 'predict_proba') else None

        # Create results DataFrame
        results = data.copy()
        results['Predicted_Risk'] = predictions
        results['Risk_Level'] = ['High Risk' if pred == 1 else 'Low Risk' for pred in predictions]

        if probabilities is not None:
            results['Probability_Low_Risk'] = probabilities[:, 0]
            results['Probability_High_Risk'] = probabilities[:, 1]

        return results

def create_sample_patient():
    """Create a sample patient for testing"""
    sample_patient = {
        'Age': 45,
        'Height_cm': 175,
        'Weight_kg': 80,
        'BMI': 26.1,
        'Systolic_BP': 130,
        'Diastolic_BP': 85,
        'Heart_Rate': 75,
        'Temperature_F': 98.6,
        'Blood_Sugar': 110,
        'Cholesterol': 220,
        'Hemoglobin': 14.5,
        'Exercise_Hours_Week': 3,
        'Hospital_Visits_Year': 1,
        'Gender': 'Male',
        'Smoking': 'No',
        'Alcohol_Consumption': 'Yes',
        'Diabetes': 'No',
        'Hypertension': 'Yes',
        'Heart_Disease': 'No',
        'Insurance_Type': 'Private',
        'Tumor_Size_cm':4.5
    }
    return sample_patient

def demonstrate_inference():
    """Demonstrate model inference capabilities"""
    inference = ModelInference()

    # Single patient prediction
    print("Single Patient Prediction Demo:")
    print("-" * 40)
    sample_patient = create_sample_patient()

    print("Patient Data:")
    for key, value in sample_patient.items():
        print(f"  {key}: {value}")

    result = inference.predict_single(sample_patient)
    print(f"\nPrediction Results:")
    print(f"  Risk Level: {result['risk_level']}")
    if result['probability_high_risk'] is not None:
        print(f"  High Risk Probability: {result['probability_high_risk']:.3f}")
        print(f"  Low Risk Probability: {result['probability_low_risk']:.3f}")

    # Batch prediction demo (if test data exists)
    try:
        print(f"\nBatch Prediction Demo:")
        print("-" * 30)
        batch_results = inference.predict_batch('X_test.csv')
        print(f"Processed {len(batch_results)} patients")
        print(f"High Risk patients: {(batch_results['Predicted_Risk'] == 1).sum()}")
        print(f"Low Risk patients: {(batch_results['Predicted_Risk'] == 0).sum()}")

        # Save batch results
        batch_results.to_csv('batch_predictions.csv', index=False)
        print("Batch results saved to: batch_predictions.csv")

    except FileNotFoundError:
        print("X_test.csv not found, skipping batch prediction demo")

if __name__ == "__main__":
    print("Model Inference System")
    print("=" * 50)

    # Demonstrate inference capabilities
    demonstrate_inference()

    print(f"\nTo use this system in your own code:")
    print("from model_inference import ModelInference")
    print("inference = ModelInference()")
    print("result = inference.predict_single(patient_data)")