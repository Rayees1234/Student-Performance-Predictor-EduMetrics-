

# Academic performance classifier for the exams dataset.
# Rough idea: turn the CSV into a simple Pass/Fail label
# and compare a basic Logistic Regression with a Random Forest.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import warnings

# I was getting a few convergence / deprecation warnings earlier,
# so I'm just muting them here to keep the output tidy.
warnings.filterwarnings('ignore')


# STEP 1: Load the dataset

df = pd.read_csv('/mnt/user-data/uploads/StudentsPerformance.csv')
print("=" * 60)
print("STEP 1: Dataset loaded")
print(f"  Total Records: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print("=" * 60)


# STEP 2: Quick exploratory checks
==
print("\nSTEP 2: Basic EDA")
print(f"  Missing values:\n{df.isnull().sum()}")
print(f"\n  Data types:\n{df.dtypes}")
print(f"\n  Score Statistics:\n{df[['math score','reading score','writing score']].describe().round(2)}")


# STEP 3: Create Pass/Fail target variable

# Simple rule: average of the three scores >= 50 -> Pass, otherwise Fail
df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
df['result'] = df['average_score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')

pass_count = (df['result'] == 'Pass').sum()
fail_count = (df['result'] == 'Fail').sum()
print("\nSTEP 3: Target variable created (avg score >= 50 -> Pass)")
print(f"  Pass: {pass_count} ({pass_count/len(df)*100:.1f}%)")
print(f"  Fail: {fail_count} ({fail_count/len(df)*100:.1f}%)")


# STEP 4: Encode categorical variables

# Turn the main categorical columns into numeric codes so the models can use them.
feature_cols = ['gender', 'race/ethnicity', 'parental level of education',
                'lunch', 'test preparation course']

df_encoded = df.copy()
encoders = {}
for col in feature_cols:
    le = LabelEncoder()
    df_encoded[col + '_enc'] = le.fit_transform(df[col])
    encoders[col] = le

print("\nSTEP 4: CATEGORICAL ENCODING (LabelEncoder)")
for col in feature_cols:
    mapping = dict(zip(encoders[col].classes_, encoders[col].transform(encoders[col].classes_)))
    print(f"  {col}: {mapping}")


# STEP 5: Prepare features (X) and target (y)

X = df_encoded[[col + '_enc' for col in feature_cols]]
y = (df_encoded['result'] == 'Pass').astype(int)  # 1=Pass, 0=Fail

print("\nSTEP 5: Features and target prepared")
print(f"  Feature columns: {list(X.columns)}")
print(f"  Target: 1=Pass, 0=Fail")


# STEP 6: Train-test split (80% train, 20% test)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("\nSTEP 6: Train-test split (80% / 20%)")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing  samples: {len(X_test)}")


# STEP 7: Train two models

# Model A: Logistic Regression (simple linear baseline)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Model B: Random Forest (non-linear, usually a bit stronger)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nSTEP 7: Model training")
print(f"  Logistic Regression Accuracy : {lr_acc*100:.2f}%")
print(f"  Random Forest Accuracy       : {rf_acc*100:.2f}%")

best_model = rf_model if rf_acc >= lr_acc else lr_model
best_pred = rf_pred if rf_acc >= lr_acc else lr_pred
best_name = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
best_acc = max(rf_acc, lr_acc)

print(f"\nBest model: {best_name} ({best_acc*100:.2f}%)")


# STEP 8: Evaluation metrics

print("\nSTEP 8: Detailed evaluation for the selected model")
print(classification_report(y_test, best_pred, target_names=['Fail', 'Pass']))


# STEP 9: Visualizations (saved as one figure)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Academic Performance Classification - Full Report", 
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

# Plot 1: Pass/Fail distribution
ax1 = fig.add_subplot(gs[0, 0])
colors = ['#e74c3c', '#2ecc71']
ax1.pie([fail_count, pass_count], labels=['Fail', 'Pass'], colors=colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize':11})
ax1.set_title('Pass / Fail distribution', fontweight='bold')

# Plot 2: Average score distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df[df['result']=='Pass']['average_score'], bins=20, alpha=0.7, color='#2ecc71', label='Pass')
ax2.hist(df[df['result']=='Fail']['average_score'], bins=20, alpha=0.7, color='#e74c3c', label='Fail')
ax2.axvline(50, color='black', linestyle='--', label='Threshold (50)')
ax2.set_xlabel('Average Score')
ax2.set_ylabel('Count')
ax2.set_title('Score distribution by result', fontweight='bold')
ax2.legend(fontsize=9)

# Plot 3: Model accuracy comparison
ax3 = fig.add_subplot(gs[0, 2])
models = ['Logistic\nRegression', 'Random\nForest']
accs = [lr_acc*100, rf_acc*100]
bars = ax3.bar(models, accs, color=['#3498db', '#e67e22'], width=0.5)
for bar, acc in zip(bars, accs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()-2,
             f'{acc:.1f}%', ha='center', va='top', color='white', fontweight='bold')
ax3.set_ylim(60, 100)
ax3.set_ylabel('Accuracy (%)')
ax3.set_title('Model accuracy comparison', fontweight='bold')

# Plot 4: Confusion matrix
ax4 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Pass'])
disp.plot(ax=ax4, colorbar=False, cmap='Blues')
ax4.set_title(f'Confusion matrix\n({best_name})', fontweight='bold')

# Plot 5: Pass rate by test prep
ax5 = fig.add_subplot(gs[1, 1])
prep_pass_rate = df.groupby('test preparation course')['result'].apply(
    lambda x: (x == 'Pass').mean() * 100).reset_index()
prep_pass_rate.columns = ['Preparation', 'Pass Rate %']
ax5.bar(prep_pass_rate['Preparation'], prep_pass_rate['Pass Rate %'], color=['#95a5a6', '#27ae60'])
ax5.set_ylabel('Pass Rate (%)')
ax5.set_title('Pass rate by test preparation', fontweight='bold')
ax5.set_ylim(0, 100)
for i, val in enumerate(prep_pass_rate['Pass Rate %']):
    ax5.text(i, val+1, f'{val:.1f}%', ha='center', fontsize=10)

# Plot 6: Pass rate by lunch type
ax6 = fig.add_subplot(gs[1, 2])
lunch_pass_rate = df.groupby('lunch')['result'].apply(
    lambda x: (x == 'Pass').mean() * 100).reset_index()
lunch_pass_rate.columns = ['Lunch', 'Pass Rate %']
ax6.bar(lunch_pass_rate['Lunch'], lunch_pass_rate['Pass Rate %'], color=['#e74c3c', '#3498db'])
ax6.set_ylabel('Pass Rate (%)')
ax6.set_title('Pass rate by lunch type', fontweight='bold')
ax6.set_ylim(0, 100)
for i, val in enumerate(lunch_pass_rate['Pass Rate %']):
    ax6.text(i, val+1, f'{val:.1f}%', ha='center', fontsize=10)

# Plot 7: Feature Importance (Random Forest)
ax7 = fig.add_subplot(gs[2, 0:2])
feat_names = ['Gender', 'Race/Ethnicity', 'Parental\nEducation', 'Lunch', 'Test Prep']
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
ax7.barh([feat_names[i] for i in sorted_idx], [importances[i] for i in sorted_idx],
         color='#8e44ad')
ax7.set_xlabel('Importance Score')
ax7.set_title('Feature Importance (Random Forest)', fontweight='bold')
ax7.invert_yaxis()

# Plot 8: Pass rate by parental education
ax8 = fig.add_subplot(gs[2, 2])
edu_pass_rate = df.groupby('parental level of education')['result'].apply(
    lambda x: (x == 'Pass').mean() * 100).sort_values()
ax8.barh(edu_pass_rate.index, edu_pass_rate.values, color='#16a085')
ax8.set_xlabel('Pass Rate (%)')
ax8.set_title('Pass rate by\nparental education', fontweight='bold')

plt.savefig('/mnt/user-data/outputs/model_report.png', dpi=150, bbox_inches='tight')
print("\nSTEP 9: Visualization saved as model_report.png")


# STEP 10: Simple sample prediction

print("\nSTEP 10: Sample prediction demo")
sample = pd.DataFrame({
    'gender_enc': [0],                        # female
    'race/ethnicity_enc': [2],                # group C
    'parental level of education_enc': [0],   # associate's degree
    'lunch_enc': [1],                         # standard
    'test preparation course_enc': [1],       # completed
})
prediction = rf_model.predict(sample)
probability = rf_model.predict_proba(sample)
print(f"  Input: female, group C, associate's degree, standard lunch, completed prep")
print(f"  Prediction: {'PASS' if prediction[0]==1 else 'FAIL'}")
print(f"  Confidence: Pass={probability[0][1]*100:.1f}%  Fail={probability[0][0]*100:.1f}%")

print("\n" + "=" * 60)
print("Model run complete")
print(f"  Best Model: {best_name}")
print(f"  Final Accuracy: {best_acc*100:.2f}%")
print("=" * 60)
