import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np

st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #0a0a0a; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; }
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #ffffff; letter-spacing: -1px; }
.metric-label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.metric-delta { font-size: 11px; color: #ef4444; margin-top: 4px; }
.section-title { font-size: 16px; font-weight: 700; color: #ffffff; margin-bottom: 4px; letter-spacing: -0.5px; }
.section-sub { font-size: 12px; color: #6b7280; margin-bottom: 1rem; }
.insight-box {
    background: rgba(220,38,38,0.06);
    border: 1px solid rgba(220,38,38,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 12px;
    color: #fca5a5;
    line-height: 1.6;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect("churn.db")
    summary = pd.read_sql_query("SELECT * FROM revenue_summary", conn)
    by_contract = pd.read_sql_query("SELECT * FROM churn_by_contract", conn)
    by_tenure = pd.read_sql_query("SELECT * FROM churn_by_tenure", conn)
    scored = pd.read_sql_query("SELECT * FROM customers_scored", conn)
    conn.close()
    return summary, by_contract, by_tenure, scored

summary, by_contract, by_tenure, scored = load_data()

# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem">
    <div style="font-size:10px;color:#ef4444;text-transform:uppercase;
                letter-spacing:2px;font-weight:600;margin-bottom:6px">
        Customer Intelligence · Telco Dataset
    </div>
    <div style="font-size:2rem;font-weight:800;color:#ffffff;letter-spacing:-1px;line-height:1.1">
        Churn Prediction Dashboard
    </div>
    <div style="font-size:13px;color:#6b7280;margin-top:6px">
        XGBoost model · AUC-ROC 0.838 · 7,043 customers · SQL + Python + Streamlit
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ─────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
s = summary.iloc[0]

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{int(s['total_customers']):,}</div>
        <div class="metric-label">Total Customers</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{int(s['total_churned']):,}</div>
        <div class="metric-label">Churned Customers</div>
        <div class="metric-delta">↑ {s['overall_churn_rate']}% churn rate</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${s['monthly_revenue_at_risk']:,.0f}</div>
        <div class="metric-label">Monthly Revenue at Risk</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${s['annual_revenue_at_risk']/1e6:.2f}M</div>
        <div class="metric-label">Annual Revenue at Risk</div>
        <div class="metric-delta">↑ Critical priority</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── ROW 1: CONTRACT + TENURE ──────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-title">Churn Rate by Contract Type</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Month-to-month customers churn at 15x the rate of two-year contracts</div>', unsafe_allow_html=True)

    fig = px.bar(
        by_contract,
        x='churn_rate_pct',
        y='contract',
        orientation='h',
        color='churn_rate_pct',
        color_continuous_scale=[[0,'#1f2937'],[0.5,'#dc2626'],[1,'#ef4444']],
        text='churn_rate_pct',
        hover_data=['total_customers', 'churned', 'monthly_revenue_at_risk']
    )
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font_color='#9ca3af',
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(title='Churn Rate (%)', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title=''),
        margin=dict(l=0, r=40, t=10, b=0),
        height=250
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
        Business insight: Converting month-to-month customers to annual contracts
        could reduce churn by up to 31 percentage points per customer.
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-title">Churn Rate by Customer Tenure</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Nearly half of new customers churn within the first 12 months</div>', unsafe_allow_html=True)

    tenure_order = ['0-12 months', '13-24 months', '25-48 months', '49+ months']
    by_tenure['tenure_group'] = pd.Categorical(
        by_tenure['tenure_group'], categories=tenure_order, ordered=True
    )
    by_tenure_sorted = by_tenure.sort_values('tenure_group')

    fig2 = px.bar(
        by_tenure_sorted,
        x='tenure_group',
        y='churn_rate_pct',
        color='churn_rate_pct',
        color_continuous_scale=[[0,'#1f2937'],[0.5,'#dc2626'],[1,'#ef4444']],
        text='churn_rate_pct',
        hover_data=['total_customers', 'churned']
    )
    fig2.update_traces(texttemplate='%{text}%', textposition='outside')
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font_color='#9ca3af',
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(title='Tenure Group', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Churn Rate (%)', gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=0, r=20, t=10, b=0),
        height=250
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
        Business insight: First-year customers are 5x more likely to churn.
        Targeted onboarding programs for months 1-12 would have the highest ROI.
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── ROW 2: RISK SEGMENTS + HIGH RISK TABLE ────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-title">Customer Risk Segments</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Model-predicted churn probability distribution</div>', unsafe_allow_html=True)

    risk_counts = scored['risk_segment'].value_counts().reset_index()
    risk_counts.columns = ['segment', 'count']
    risk_order = ['High Risk', 'Medium Risk', 'Low Risk']
    risk_counts['segment'] = pd.Categorical(
        risk_counts['segment'], categories=risk_order, ordered=True
    )
    risk_counts = risk_counts.sort_values('segment')
    colors = {'High Risk': '#ef4444', 'Medium Risk': '#f97316', 'Low Risk': '#22c55e'}

    fig3 = px.bar(
        risk_counts,
        x='segment',
        y='count',
        color='segment',
        color_discrete_map=colors,
        text='count'
    )
    fig3.update_traces(textposition='outside')
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font_color='#9ca3af',
        showlegend=False,
        xaxis=dict(title='', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Number of Customers', gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=0, r=20, t=10, b=0),
        height=280
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.markdown('<div class="section-title">Top High Risk Customers</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Sorted by churn probability — prioritise for retention outreach</div>', unsafe_allow_html=True)

    high_risk = scored[scored['risk_segment'] == 'High Risk'].nlargest(10, 'churn_probability')
    display_cols = ['customerid', 'contract', 'tenure', 'monthlycharges', 'churn_probability']
    display = high_risk[display_cols].copy()
    display['churn_probability'] = (display['churn_probability'] * 100).round(1).astype(str) + '%'
    display['monthlycharges'] = '$' + display['monthlycharges'].round(2).astype(str)
    display.columns = ['Customer ID', 'Contract', 'Tenure', 'Monthly $', 'Churn Prob']
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=280
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── ROW 3: SHAP + REVENUE GAUGE ───────────────────────────────
col5, col6 = st.columns(2)

with col5:
    st.markdown('<div class="section-title">Top Churn Drivers — SHAP Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Features with highest impact on churn prediction</div>', unsafe_allow_html=True)
    st.image('shap_importance.png', use_container_width=True)

with col6:
    st.markdown('<div class="section-title">Revenue at Risk by Contract</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Monthly revenue at risk from churning customers per contract type</div>', unsafe_allow_html=True)

    fig5 = px.pie(
        by_contract,
        values='monthly_revenue_at_risk',
        names='contract',
        color_discrete_sequence=['#ef4444', '#f97316', '#6b7280'],
        hole=0.6
    )
    fig5.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#9ca3af',
        legend=dict(font=dict(color='#9ca3af')),
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
        annotations=[dict(
            text=f"${s['monthly_revenue_at_risk']/1e3:.0f}K<br>at risk",
            x=0.5, y=0.5,
            font_size=14,
            font_color='#ffffff',
            showarrow=False
        )]
    )
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── ROW 4: CHURN PROBABILITY CHECKER ─────────────────────────
st.markdown('<div class="section-title">Individual Customer Churn Probability Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Enter customer details to get real-time churn probability from the model</div>', unsafe_allow_html=True)

with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']

c1, c2, c3, c4 = st.columns(4)
with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
with c2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
with c3:
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
with c4:
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

if st.button("Calculate Churn Probability", type="primary"):
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    payment_map = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }
    yn = lambda x: 1 if x == "Yes" else 0

    total_charges = tenure * monthly_charges
    charges_per_tenure = monthly_charges / (tenure + 1)

    input_data = {
        'gender': 0, 'seniorcitizen': 0, 'partner': 0, 'dependents': 0,
        'tenure': tenure, 'phoneservice': 1,
        'multiplelines': 0,
        'internetservice': internet_map[internet],
        'onlinesecurity': yn(online_security),
        'onlinebackup': 0,
        'deviceprotection': 0,
        'techsupport': yn(tech_support),
        'streamingtv': 0, 'streamingmovies': 0,
        'contract': contract_map[contract],
        'paperlessbilling': yn(paperless),
        'paymentmethod': payment_map[payment],
        'monthlycharges': monthly_charges,
        'totalcharges': total_charges,
        'churn_binary': 0,
        'charges_per_tenure': charges_per_tenure,
        'total_services': yn(online_security) + yn(tech_support) + 1,
        'is_high_value': 1 if monthly_charges > 64.76 else 0
    }

    input_df = pd.DataFrame([input_data])[features]
    prob = model.predict_proba(input_df)[0][1]

    color = "#ef4444" if prob > 0.6 else "#f97316" if prob > 0.3 else "#22c55e"
    risk_label = "High Risk" if prob > 0.6 else "Medium Risk" if prob > 0.3 else "Low Risk"

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                border-radius:12px;padding:1.5rem;margin-top:1rem;display:flex;
                align-items:center;gap:2rem">
        <div style="text-align:center;min-width:140px">
            <div style="font-size:3rem;font-weight:800;color:{color}">{prob*100:.1f}%</div>
            <div style="font-size:12px;color:{color};font-weight:600;
                        background:rgba(255,255,255,0.05);padding:4px 16px;
                        border-radius:20px;margin-top:6px;border:1px solid {color}">
                {risk_label}
            </div>
        </div>
        <div style="font-size:13px;color:#9ca3af;line-height:1.8">
            <strong style="color:#ffffff">Key factors:</strong><br>
            Contract: <strong style="color:#ffffff">{contract}</strong> 
            ({"highest" if contract == "Month-to-month" else "lower"} risk)<br>
            Tenure: <strong style="color:#ffffff">{tenure} months</strong>
            ({"early stage — high risk" if tenure < 12 else "established customer"})<br>
            Monthly charges: <strong style="color:#ffffff">${monthly_charges}</strong><br>
            Charges per tenure month: <strong style="color:#ffffff">${charges_per_tenure:.2f}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)