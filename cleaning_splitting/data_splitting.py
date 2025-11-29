import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv('processed/fair_salary_prediction_dataset.csv')
print(f"Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Columns to exclude from features
exclude_cols = [
    'Applicant_ID', 'Expected_CTC', 'Education', 'Experience_Band',
    'Education_Level', 'Department', 'Role', 'Industry', 'Organization',
    'Designation', 'Graduation_Specialization', 'University_Grad', 
    'PG_Specialization', 'University_PG', 'PHD_Specialization', 
    'University_PHD', 'Curent_Location', 'Preferred_location',
    'Last_Appraisal_Rating'
]

# Select features: encoded columns + engineered numeric features
feature_cols = [
    col for col in df.columns 
    if (
        '_encoded' in col or col in [
            'Total_Experience', 'Total_Experience_in_field_applied',
            'Current_CTC', 'No_Of_Companies_worked', 'Number_of_Publications',
            'Certifications', 'Inhand_Offer', 'International_degree_any',
            'Has_PG', 'Has_PHD', 'Performance_Score', 'Field_Specialization_Ratio'
        ]
    ) and col not in exclude_cols
]

X = df[feature_cols]
y = df['Expected_CTC']

print(f"Selected {len(feature_cols)} features for prediction")
print("Target variable: Expected_CTC")

# Perform an 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"Split complete: {len(X_train)} training samples, {len(X_test)} testing samples")

# Save the splits
X_train.to_csv('processed/X_train.csv', index=False)
X_test.to_csv('processed/X_test.csv', index=False)
y_train.to_csv('processed/y_train.csv', index=False, header=True)
y_test.to_csv('processed/y_test.csv', index=False, header=True)

# Summary
print("\nSplit Summary:")
print(f"Features shape: {X.shape}")
print(f"Training features: {X_train.shape}")
print(f"Test features: {X_test.shape}")
print(f"Training target range: ₹{y_train.min():,.0f} - ₹{y_train.max():,.0f}")
print(f"Test target range: ₹{y_test.min():,.0f} - ₹{y_test.max():,.0f}")

print("\nData is ready for model training. Next, run 'src/model_training.py'.")
