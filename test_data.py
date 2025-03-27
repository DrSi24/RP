import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of test records
n_records = 500

# Define all possible values for categorical variables
healthcare_roles = ["Doctor", "Nurse", "Admin Staff", "Paramedic", "Technician", 
                   "Therapist", "Pharmacist", "Midwife", "Healthcare Assistant"]
                   
departments = ["Emergency", "Surgery", "Pediatrics", "Oncology", "Cardiology", 
              "Neurology", "Psychiatry", "Radiology", "General Practice", 
              "Intensive Care", "Obstetrics", "Geriatrics", "Administration"]

mh_disorders_list = ["None", "Depression", "Anxiety", "Bipolar", "PTSD", 
                    "Depression, Anxiety", "Multiple"]

substance_disorders_list = ["None", "Alcohol", "Cannabis", "Prescription Drugs", 
                          "Multiple", "None", "None"]

chronic_illnesses_list = ["None", "Diabetes", "Hypertension", "Asthma", 
                         "Multiple", "None", "None"]

# Generate base data
data = {
    # Demographic & Socioeconomic
    'Age': np.random.randint(22, 71, n_records),
    'Sex': np.random.choice(['Male', 'Female', 'Other'], n_records, p=[0.45, 0.50, 0.05]),
    'Employment_Status': np.random.choice(['Full-time', 'Part-time', 'Contract', 'Temporary'], n_records),
    'Income_Level': np.random.choice(['Low', 'Medium', 'High'], n_records),
    'Social_Deprivation': np.random.randint(0, 11, n_records),
    'Material_Deprivation': np.random.randint(0, 11, n_records),

    # Occupational Data (New Section)
    'Healthcare_Role': np.random.choice(healthcare_roles, n_records),
    'Department': np.random.choice(departments, n_records),
    'Years_Experience': np.random.randint(0, 41, n_records),
    'Weekly_Hours': np.random.randint(20, 61, n_records),
    'Night_Shifts_Monthly': np.random.randint(0, 13, n_records),
    'Overtime_Hours_Monthly': np.random.randint(0, 41, n_records),
    'Patient_Facing': np.random.choice(['Yes', 'No'], n_records),
    'Management_Responsibilities': np.random.choice(['Yes', 'No'], n_records),
    'Work_Stress_Level': np.random.randint(0, 11, n_records),
    'Job_Satisfaction': np.random.randint(0, 11, n_records),
    'Workplace_Support': np.random.randint(0, 11, n_records),
    'Burnout_Level': np.random.randint(0, 11, n_records),
    'Sick_Days_Last_Year': np.random.randint(0, 31, n_records),
    'Workplace_Incidents': np.random.randint(0, 6, n_records),
    'Recent_Promotion': np.random.choice(['Yes', 'No'], n_records, p=[0.2, 0.8]),
    'Recent_Demotion': np.random.choice(['Yes', 'No'], n_records, p=[0.05, 0.95]),

    # Clinical & Psychiatric
    'MH_Disorders': [random.choice(mh_disorders_list) for _ in range(n_records)],
    'Substance_Use_Disorders': [random.choice(substance_disorders_list) for _ in range(n_records)],
    'History_Suicidal_Ideation': np.random.choice([0, 1], n_records, p=[0.8, 0.2]),
    'Previous_Suicide_Attempts': np.random.choice([0, 1, 2, 3], n_records, p=[0.85, 0.1, 0.03, 0.02]),
    'Frequency_Suicidal_Thoughts': np.random.randint(0, 11, n_records),
    'Intensity_Suicidal_Thoughts': np.random.randint(0, 11, n_records),

    # Health & Medical
    'Chronic_Illnesses': [random.choice(chronic_illnesses_list) for _ in range(n_records)],
    'GP_Visits': np.random.randint(0, 21, n_records),
    'ED_Visits': np.random.randint(0, 11, n_records),
    'Hospitalizations': np.random.randint(0, 6, n_records),

    # Psychological Factors
    'Hopelessness': np.random.randint(0, 11, n_records),
    'Despair': np.random.randint(0, 11, n_records),
    'Impulsivity': np.random.randint(0, 11, n_records),
    'Aggression': np.random.randint(0, 11, n_records),
    'Access_Lethal_Means': np.random.randint(0, 11, n_records),
    'Social_Isolation': np.random.randint(0, 11, n_records),

    # Support & Resilience
    'Coping_Strategies': np.random.randint(0, 11, n_records),
    'Measured_Resilience': np.random.randint(0, 11, n_records),
    'MH_Service_Engagement': np.random.randint(0, 11, n_records),
    'Supportive_Relationships': np.random.randint(0, 11, n_records),
}

# Create DataFrame
df = pd.DataFrame(data)

# Add correlated and derived fields
# Correlate burnout with work stress and overtime
df['Burnout_Level'] = np.clip(
    df['Work_Stress_Level'] * 0.4 + 
    df['Overtime_Hours_Monthly'] * 0.1 + 
    np.random.normal(0, 1, n_records),
    0, 10
).astype(int)

# Correlate job satisfaction inversely with burnout
df['Job_Satisfaction'] = np.clip(
    10 - df['Burnout_Level'] * 0.7 + 
    np.random.normal(0, 1, n_records),
    0, 10
).astype(int)

# Calculate Suicidal_Distress as weighted average of risk factors
df['Suicidal_Distress'] = np.clip(
    0.2 * df['Hopelessness'] +
    0.2 * df['Despair'] +
    0.15 * df['Social_Isolation'] +
    0.15 * df['Burnout_Level'] +
    0.1 * df['Work_Stress_Level'] +
    0.1 * df['Access_Lethal_Means'] -
    0.1 * df['Supportive_Relationships'] -
    0.1 * df['Job_Satisfaction'] +
    np.random.normal(0, 1, n_records),
    0, 10
).astype(int)

# Calculate Time_To_Crisis (days) - higher risk scores have shorter times
df['Time_To_Crisis'] = np.clip(
    365 * (1 - df['Suicidal_Distress']/15) + 
    np.random.normal(0, 30, n_records),
    7, 365
).astype(int)

# Crisis event (more likely with higher distress)
crisis_prob = df['Suicidal_Distress'] / 15
df['Crisis_Event'] = np.random.binomial(1, crisis_prob)

# Generate observation dates (within the last year)
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
random_days = np.random.randint(0, 365, n_records)
df['Observation_Date'] = [(start_date + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S") 
                         for days in random_days]

# Save to CSV
df.to_csv('resume_test_data_full.csv', index=False)
print("Test data CSV created successfully with", n_records, "records!")

# Display some basic statistics
print("\nBasic Statistics:")
print(f"Total Records: {len(df)}")
print(f"Crisis Events: {df['Crisis_Event'].sum()}")
print(f"Average Burnout Level: {df['Burnout_Level'].mean():.2f}")
print(f"Average Suicidal Distress: {df['Suicidal_Distress'].mean():.2f}")
