from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import json
import io
import base64
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry
import geonamescache

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'model.pkl')
METADATA_PATH = os.path.join(BASE_DIR, '..', 'results', 'model_metadata.json')
FEATURE_IMP_PATH = os.path.join(BASE_DIR, '..', 'results', 'feature_importance.csv')
X_TRAIN_PATH = os.path.join(BASE_DIR, '..', 'processed', 'X_train.csv')
Y_TRAIN_PATH = os.path.join(BASE_DIR, '..', 'processed', 'y_train.csv')
FULL_DATA_PATH = os.path.join(BASE_DIR, '..', 'processed', 'fair_salary_prediction_dataset.csv')

# Load model and data
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        feature_imp = pd.read_csv(FEATURE_IMP_PATH)
        train_data = pd.read_csv(X_TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH)
        full_data = pd.read_csv(FULL_DATA_PATH)
        return model, metadata, feature_imp, train_data, y_train, full_data
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

model, metadata, feature_importance, train_data, y_train, full_dataset = load_resources()

# Helper function for matplotlib
def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img

#  Routes  

@app.route('/')
def dashboard():
    try:
        stats = {
            'total': len(full_dataset),
            'avg_salary': f"{full_dataset['Expected_CTC'].mean():,.0f}" if 'Expected_CTC' in full_dataset.columns else "N/A",
            'model_loaded': model is not None
        }

        # Salary distribution
        plt.figure(figsize=(10, 5))
        if 'Expected_CTC' in full_dataset.columns:
            plt.hist(full_dataset['Expected_CTC'].dropna(), bins=30, color='#4f46e5', alpha=0.7, edgecolor='black')
            plt.xlabel('Salary (CTC)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Salary Distribution', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
        chart1 = fig_to_base64()

        # Department distribution
        plt.figure(figsize=(10, 5))
        if 'Department' in full_dataset.columns:
            dept = full_dataset['Department'].value_counts().head(8)
            dept.plot(kind='barh', color='#10b981')
            plt.xlabel('Count', fontsize=12)
            plt.title('Top Departments', fontsize=14, fontweight='bold')
            plt.tight_layout()
        chart2 = fig_to_base64()

        return render_template('dashboard.html', stats=stats, chart1=chart1, chart2=chart2)
    except Exception as e:
        return render_template('dashboard.html', error=str(e))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Pass dropdowns except locations
        dropdowns = {
            'departments': sorted(full_dataset['Department'].dropna().unique().tolist()) if 'Department' in full_dataset.columns else [],
            'roles': sorted(full_dataset['Role'].dropna().unique().tolist()) if 'Role' in full_dataset.columns else [],
            'industries': sorted(full_dataset['Industry'].dropna().unique().tolist()) if 'Industry' in full_dataset.columns else []
        }

        if request.method == 'GET':
            return render_template('predict.html', **dropdowns)

        # POST - single prediction
        data = {k: v for k, v in request.form.items()}
        df = pd.DataFrame([data])

        # Numeric conversion
        numeric_cols = [
            'Total_Experience', 'Total_Experience_in_field_applied', 'Current_CTC',
            'Inhand_Offer', 'Last_Appraisal_Rating', 'No_Of_Companies_worked',
            'Number_of_Publications', 'Certifications', 'Passing_Year_Of_Graduation',
            'Passing_Year_Of_PG', 'Passing_Year_Of_PHD'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Align columns with training
        for col in train_data.columns:
            if col not in df.columns:
                df[col] = 0
        df = df[train_data.columns]

        pred = model.predict(df)[0]
        result = {
            'predicted': f"{pred:,.2f}",
            'lower': f"{pred*0.9:,.2f}",
            'upper': f"{pred*1.1:,.2f}"
        }

        # Feature importance (top 5)
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            top_feat = sorted(zip(train_data.columns, imp), key=lambda x: x[1], reverse=True)[:5]
            result['features'] = [(f, f"{v:.4f}") for f, v in top_feat]

        return render_template('predict.html', result=result, **dropdowns)
    except Exception as e:
        return render_template('predict.html', error=str(e))

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'GET':
        return render_template('batch.html')

    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('batch.html', error="No file uploaded")

        try:
            df = pd.read_csv(file, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(file, encoding='latin1')
            except:
                df = pd.read_csv(file, engine='python', on_bad_lines='skip')

        # Numeric conversion
        numeric_cols = [
            'Total_Experience', 'Total_Experience_in_field_applied', 'Current_CTC', 'Inhand_Offer',
            'Last_Appraisal_Rating', 'No_Of_Companies_worked', 'Number_of_Publications',
            'Certifications', 'Passing_Year_Of_Graduation', 'Passing_Year_Of_PG', 'Passing_Year_Of_PHD'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

        # International degree
        if 'International_degree_any' in df.columns:
            df['International_degree_any'] = df['International_degree_any'].map({'Y': 1, 'N': 0}).fillna(0)

        applicant_ids = df.get('Applicant_ID', '').copy()

        # One-hot encode categorical
        categorical_cols = [
            'Department', 'Role', 'Industry', 'Current_Location', 'Preferred_location',
            'Education', 'Graduation_Specialization', 'PG_Specialization',
            'PHD_Specialization', 'Organization', 'Designation'
        ]
        for cat_col in categorical_cols:
            encoded_cols = [c for c in train_data.columns if c.startswith(cat_col + '_')]
            for encoded in encoded_cols:
                value = encoded.replace(cat_col + '_', '')
                df[encoded] = (df.get(cat_col, '') == value).astype(int)

        # Add missing columns
        for col in train_data.columns:
            if col not in df.columns:
                df[col] = 0

        df = df[train_data.columns]

        predictions = model.predict(df)
        results = pd.DataFrame({
            'Applicant_ID': applicant_ids,
            'Predicted_Salary': predictions,
            'Range_Lower': (predictions * 0.9).round(2),
            'Range_Upper': (predictions * 1.1).round(2)
        })

        summary = {
            'count': len(predictions),
            'mean': f"{np.mean(predictions):,.2f}",
            'median': f"{np.median(predictions):,.2f}",
            'min': f"{np.min(predictions):,.2f}",
            'max': f"{np.max(predictions):,.2f}"
        }

        plt.figure(figsize=(10, 5))
        plt.hist(predictions, bins=30, color='#8b5cf6', alpha=0.7, edgecolor='black')
        plt.xlabel('Predicted Salary')
        plt.ylabel('Frequency')
        plt.title('Batch Prediction Salary Distribution')
        chart = fig_to_base64()

        preview = results.head(50).to_dict('records')
        results_csv = results.to_csv(index=False)
        download_b64 = base64.b64encode(results_csv.encode()).decode()

        return render_template('batch.html', summary=summary, chart=chart, preview=preview, download=download_b64)

    except Exception as e:
        return render_template('batch.html', error=str(e))

@app.route('/analytics')
def analytics():
    try:
        plt.figure(figsize=(10, 6))
        if not feature_importance.empty:
            top = feature_importance.head(15)
            plt.barh(top['Feature'], top['Average_Importance'], color='#f59e0b')
            plt.xlabel('Importance')
            plt.title('Top 15 Feature Importance')
            plt.tight_layout()
        chart1 = fig_to_base64()

        plt.figure(figsize=(10, 6))
        if 'Department' in full_dataset.columns and 'Expected_CTC' in full_dataset.columns:
            dept_salary = full_dataset.groupby('Department')['Expected_CTC'].mean().sort_values(ascending=False).head(10)
            dept_salary.plot(kind='barh', color='#06b6d4')
            plt.xlabel('Average Salary')
            plt.title('Average Salary by Department')
            plt.tight_layout()
        chart2 = fig_to_base64()

        plt.figure(figsize=(10, 6))
        if 'Total_Experience' in full_dataset.columns and 'Expected_CTC' in full_dataset.columns:
            sample = full_dataset.sample(min(1000, len(full_dataset)))
            plt.scatter(sample['Total_Experience'], sample['Expected_CTC'], alpha=0.5, color='#ec4899')
            plt.xlabel('Total Experience (years)')
            plt.ylabel('Expected CTC')
            plt.title('Experience vs Salary')
            plt.grid(alpha=0.3)
        chart3 = fig_to_base64()

        return render_template('analytics.html', chart1=chart1, chart2=chart2, chart3=chart3)
    except Exception as e:
        return render_template('analytics.html', error=str(e))

@app.route('/fairness')
def fairness():
    try:
        charts = []

        # Salary by Location
        fig, ax = plt.subplots(figsize=(12, 6))
        if 'Current_Location' in full_dataset.columns and 'Expected_CTC' in full_dataset.columns:
            loc_salary = full_dataset.groupby('Current_Location')['Expected_CTC'].agg(['mean', 'std']).sort_values('mean', ascending=False).head(10)
            x = range(len(loc_salary))
            ax.bar(x, loc_salary['mean'], yerr=loc_salary['std'], capsize=5, color='#3b82f6', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(loc_salary.index, rotation=45, ha='right')
            ax.set_ylabel('Average Salary')
            ax.set_title('Salary Distribution by Location')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        charts.append(base64.b64encode(buf.read()).decode('utf-8'))
        plt.close(fig)

        # Salary by Department boxplot
        fig, ax = plt.subplots(figsize=(12, 6))
        if 'Department' in full_dataset.columns and 'Expected_CTC' in full_dataset.columns:
            top_depts = full_dataset['Department'].value_counts().head(8).index
            data_subset = full_dataset[full_dataset['Department'].isin(top_depts)]
            data_subset.boxplot(column='Expected_CTC', by='Department', ax=ax)
            ax.set_ylabel('Salary')
            ax.set_title('Salary Distribution by Department')
            fig.suptitle('')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        charts.append(base64.b64encode(buf.read()).decode('utf-8'))
        plt.close(fig)

        # Salary by Education
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'Education' in full_dataset.columns and 'Expected_CTC' in full_dataset.columns and full_dataset['Education'].dropna().any():
            edu_salary = full_dataset.groupby('Education')['Expected_CTC'].mean().sort_values(ascending=False)
            edu_salary.plot(kind='barh', color='#10b981', ax=ax)
            ax.set_xlabel('Average Salary')
            ax.set_title('Average Salary by Education Level')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        charts.append(base64.b64encode(buf.read()).decode('utf-8'))
        plt.close(fig)

        return render_template('fairness.html', charts=charts)
    except Exception as e:
        return render_template('fairness.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=50011)
