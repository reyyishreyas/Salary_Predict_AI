import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('processed/fair_salary_prediction_dataset.csv')
print(f"\n Loaded: {df.shape}")

exclude = [
    'Applicant_ID', 'Expected_CTC', 'Education', 'Experience_Band',
    'Education_Level', 'Department', 'Role', 'Industry', 'Organization',
    'Designation', 'Graduation_Specialization', 'University_Grad', 
    'PG_Specialization', 'University_PG', 'PHD_Specialization', 
    'University_PHD', 'Curent_Location', 'Preferred_location',
    'Last_Appraisal_Rating'
]

feature_cols = [col for col in df.columns if (
    '_encoded' in col or
    col in [
        'Total_Experience', 'Total_Experience_in_field_applied',
        'Current_CTC', 'No_Of_Companies_worked', 'Number_of_Publications',
        'Certifications', 'Inhand_Offer', 'International_degree_any',
        'Has_PG', 'Has_PHD', 'Performance_Score', 'Field_Specialization_Ratio',
        'Passing_Year_Of_Graduation', 'Passing_Year_Of_PG', 'Passing_Year_Of_PHD'
    ]
) and col not in exclude]

X = df[feature_cols]
y = df['Expected_CTC']

print(f"Selected {len(feature_cols)} predictive features")

y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y_bins  
)

print(f" Stratified split completed:")
print(f"   Training: {len(X_train)} samples (80%)")
print(f"   Testing:  {len(X_test)} samples (20%)")

print(f" Split quality check:")
print(f"   Train salary range: ₹{y_train.min():,.0f} - ₹{y_train.max():,.0f}")
print(f"   Test salary range:  ₹{y_test.min():,.0f} - ₹{y_test.max():,.0f}")
print(f"   Train mean: ₹{y_train.mean():,.0f}")
print(f"   Test mean:  ₹{y_test.mean():,.0f}")
print(f"   Distribution similarity: {abs(y_train.mean() - y_test.mean()) / y_train.mean() * 100:.2f}% difference")

# SAVE SPLITS 
X_train.to_csv('processed/X_train.csv', index=False)
X_test.to_csv('processed/X_test.csv', index=False)
y_train.to_csv('processed/y_train.csv', index=False)
y_test.to_csv('processed/y_test.csv', index=False)

print(f"\n✓ Saved to processed/ folder")
