import pandas as pd
import numpy as np
import sqlite3
import shap
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier

DB_PATH = "churn.db"

# ── 1. Load ───────────────────────────────────────────────────
print("Loading data from SQLite...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()
print(f"Loaded {len(df)} customers")

# ── 2. Feature Engineering ────────────────────────────────────
print("Engineering features...")
df = df.drop(['customerid', 'churn'], axis=1)

binary_cols = ['gender', 'partner', 'dependents', 'phoneservice',
               'paperlessbilling', 'multiplelines', 'onlinesecurity',
               'onlinebackup', 'deviceprotection', 'techsupport',
               'streamingtv', 'streamingmovies']

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0,
                                'Male': 1, 'Female': 0,
                                'No phone service': 0,
                                'No internet service': 0})

le = LabelEncoder()
for col in ['internetservice', 'contract', 'paymentmethod']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

df['charges_per_tenure'] = df['monthlycharges'] / (df['tenure'] + 1)
df['total_services'] = df[['phoneservice','multiplelines','onlinesecurity',
                            'onlinebackup','deviceprotection','techsupport',
                            'streamingtv','streamingmovies']].sum(axis=1)
df['is_high_value'] = (df['monthlycharges'] > df['monthlycharges'].median()).astype(int)

X = df.drop('churn_binary', axis=1)
y = df['churn_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Train ──────────────────────────────────────────────────
print("Training XGBoost...")
model = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=3, random_state=42,
    eval_metric='logloss', verbosity=0
)
model.fit(X_train, y_train)

# ── 4. Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("\n── Model Performance ──")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# ── 5. ROC Curve ─────────────────────────────────────────────
print("Saving ROC curve...")
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#534AB7', lw=2, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Churn Prediction Model')
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("roc_curve.png saved!")

# ── 6. SHAP ───────────────────────────────────────────────────
print("Generating SHAP values (this takes ~30 seconds)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:500])
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test[:500], plot_type="bar", show=False)
plt.title("Top Features Driving Churn — SHAP Values")
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("shap_importance.png saved!")

# ── 7. Save predictions to DB ─────────────────────────────────
print("Saving predictions to database...")
conn = sqlite3.connect(DB_PATH)
original = pd.read_sql_query("SELECT * FROM customers", conn)
original['churn_probability'] = model.predict_proba(X)[:, 1].round(4)
original['churn_predicted'] = model.predict(X)
original['risk_segment'] = pd.cut(
    original['churn_probability'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)
original.to_sql('customers_scored', conn, if_exists='replace', index=False)
conn.commit()
conn.close()
print("Predictions saved!")

# ── 8. Save model ─────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': list(X.columns)}, f)
print("Model saved as model.pkl")
print("\nAll done!")