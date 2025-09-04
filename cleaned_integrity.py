import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()

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

    # Incorrect formats
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

    data['BMI'] = data['Weight_kg'] / ((data['Height_cm'] / 100) ** 2)
    data['Risk_Score'] = data.apply(calculate_risk, axis=1)
    return add_anomalies(data, random_state=random_state)

# -------------------------------
# Cleaning with IQR + Z-Score + Rules
# -------------------------------
def clean_dataset(df, z_thresh=3):
    print("Initial shape:", df.shape)

    # 1. Handle Missing
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
    dataset.to_csv("patient_data_with_anomalies.csv", index=False)

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

    # ✅ Ensure exactly 1000 rows (cut off extras if too many)
    cleaned = cleaned.head(TARGET_ROWS)

    print(f"✅ Final dataset has exactly {len(cleaned)} rows")
    cleaned.to_csv("patient_data_cleaned.csv", index=False)
    print("Saved cleaned dataset.")
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