import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

NUM_PATIENTS = 1000

first_names = ['Amit', 'Priya', 'Rahul', 'Sneha', 'Vikram', 'Anjali', 'Rohan', 'Neha', 'Suresh', 'Pooja']
last_names = ['Sharma', 'Patel', 'Singh', 'Gupta', 'Kumar', 'Reddy', 'Jain', 'Mehta', 'Joshi', 'Das']
genders = ['Male', 'Female', 'Other']
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
states = ['Maharashtra', 'Delhi', 'Karnataka', 'Telangana', 'Tamil Nadu', 'West Bengal', 'Gujarat', 'Rajasthan', 'Uttar Pradesh']
blood_types = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
allergy_list = ['None', 'Penicillin', 'Peanuts', 'Dust', 'Pollen', 'Latex', 'Seafood']
chronic_conditions_list = ['None', 'Diabetes', 'Hypertension', 'Asthma', 'COPD', 'Arthritis']
medications_list = ['None', 'Metformin', 'Amlodipine', 'Salbutamol', 'Atorvastatin', 'Paracetamol']
diagnosis_list = ['Healthy', 'Flu', 'Diabetes', 'Hypertension', 'Asthma', 'Allergy', 'Infection']
insurance_providers = ['ICICI Lombard', 'HDFC Ergo', 'Star Health', 'Max Bupa', 'Apollo Munich']

def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

def random_phone():
    return '+91' + ''.join(np.random.choice(list('0123456789'), 10))

def random_email(first, last, pid):
    return f"{first.lower()}.{last.lower()}{pid[-4:]}@example.com"

def random_address():
    return f"{random.randint(101,999)}, {random.choice(['MG Road', 'Park Street', 'Ring Road', 'Main Street', 'Station Road'])}"

def random_postal():
    return str(random.randint(100000, 999999))

def random_policy():
    return 'PL' + ''.join(np.random.choice(list('0123456789'), 8))

today = datetime.now()
start_birth = today - timedelta(days=365*90)
end_birth = today - timedelta(days=365*1)

data = []
for i in range(NUM_PATIENTS):
    pid = f"PT{100001 + i}"
    first = random.choice(first_names)
    last = random.choice(last_names)
    gender = random.choice(genders)
    dob = random_date(start_birth, end_birth)
    age = int((today - dob).days // 365.25)
    phone = random_phone()
    email = random_email(first, last, pid)
    address = random_address()
    city = random.choice(cities)
    state = random.choice(states)
    country = 'India'
    postal = random_postal()
    blood = random.choice(blood_types)
    allergies = random.choice(allergy_list)
    chronic = random.choice(chronic_conditions_list)
    medication = random.choice(medications_list)
    last_visit = random_date(today - timedelta(days=730), today)
    diagnosis = random.choice(diagnosis_list)
    height_cm = np.random.normal(165, 10)
    weight_kg = np.random.normal(65, 15)
    bmi = round(weight_kg / ((height_cm/100)**2), 1)
    smoker = np.random.choice([True, False], p=[0.2, 0.8])
    alcohol = np.random.choice([True, False], p=[0.3, 0.7])
    emergency_name = random.choice(first_names) + ' ' + random.choice(last_names)
    emergency_phone = random_phone()
    insurance = random.choice(insurance_providers)
    policy = random_policy()
    created_at = random_date(today - timedelta(days=365*2), today)
    updated_at = created_at + timedelta(days=random.randint(0, 365))
    data.append({
        'patient_id': pid,
        'first_name': first,
        'last_name': last,
        'gender': gender,
        'date_of_birth': dob.strftime('%Y-%m-%d'),
        'age': age,
        'phone_number': phone,
        'email_address': email,
        'address': address,
        'city': city,
        'state': state,
        'country': country,
        'postal_code': postal,
        'blood_type': blood,
        'allergies': allergies,
        'chronic_conditions': chronic,
        'current_medication': medication,
        'last_visit_date': last_visit.strftime('%Y-%m-%d'),
        'diagnosis': diagnosis,
        'weight_kg': round(weight_kg, 1),
        'height_cm': round(height_cm, 1),
        'bmi': bmi,
        'smoker': smoker,
        'alcohol_use': alcohol,
        'emergency_contact_name': emergency_name,
        'emergency_contact_phone': emergency_phone,
        'insurance_provider': insurance,
        'policy_number': policy,
        'record_created_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),
        'record_updated_at': updated_at.strftime('%Y-%m-%d %H:%M:%S')
    })

df = pd.DataFrame(data)
df.to_csv('patients.csv', index=False)
df.to_json('patients.json', orient='records', lines=False, force_ascii=False)
print("Synthetic patient data exported to patients.csv and patients.json")

# Mapping of city to correct state
city_state_map = {
    "Bangalore": "Karnataka",
    "Pune": "Maharashtra",
    "Lucknow": "Uttar Pradesh",
    "Hyderabad": "Telangana",
    "Ahmedabad": "Gujarat",
    "Delhi": "Delhi",
    "Jaipur": "Rajasthan",
    "Chennai": "Tamil Nadu",
    "Mumbai": "Maharashtra",
    "Kolkata": "West Bengal"
    # Add more if needed
}

# Read the CSV
df = pd.read_csv('patients.csv')

# Update the state column based on city
df['state'] = df['city'].map(city_state_map).fillna(df['state'])

# Save to a new CSV
df.to_csv('patients_corrected.csv', index=False)