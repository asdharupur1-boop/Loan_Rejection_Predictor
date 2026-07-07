# app.py
# ============================================================================
# LOAN REJECTION PREDICTION & RISK ANALYTICS SYSTEM
# IIT Jammu - Internship Program | Week 5 Assignment
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
import os
import base64
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Loan Rejection Prediction - IIT Jammu",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS WITH IIT JAMMU BRANDING
# ============================================================================
st.markdown("""
<style>
    /* Main header with IIT Jammu colors */
    .main-header {
        font-size: 2.5rem;
        color: #1a237e;
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #e8eaf6, #c5cae9, #e8eaf6);
        border-radius: 10px;
        margin-bottom: 2rem;
        border-bottom: 4px solid #1a237e;
    }
    
    /* IIT Jammu branding bar */
    .iit-header {
        background: linear-gradient(90deg, #1a237e, #283593, #1a237e);
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 1rem;
        box-shadow: 0 2px 10px rgba(26, 35, 126, 0.3);
    }
    
    .iit-header a {
        color: white;
        text-decoration: none;
        margin: 0 12px;
        padding: 4px 10px;
        border-radius: 5px;
        transition: background-color 0.3s;
        font-size: 0.9rem;
    }
    
    .iit-header a:hover {
        background-color: rgba(255, 255, 255, 0.2);
        text-decoration: none;
    }
    
    .iit-header .separator {
        color: rgba(255,255,255,0.3);
        margin: 0 8px;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #1a237e;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1a237e;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 5px;
    }
    
    /* Risk indicators */
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
        padding: 2px 10px;
        border-radius: 4px;
        background: #fde8e8;
        display: inline-block;
    }
    
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
        padding: 2px 10px;
        border-radius: 4px;
        background: #fef3e0;
        display: inline-block;
    }
    
    .risk-low {
        color: #2ecc71;
        font-weight: bold;
        padding: 2px 10px;
        border-radius: 4px;
        background: #e8f8ed;
        display: inline-block;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #1a237e, #283593);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(26, 35, 126, 0.4);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        border-top: 1px solid #ddd;
        color: #666;
        font-size: 0.85rem;
    }
    
    .footer a {
        color: #1a237e;
        text-decoration: none;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Sidebar */
    .sidebar-logo {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }
    
    .sidebar-logo img {
        max-width: 70px;
    }
    
    .sidebar-title {
        color: #1a237e;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0;
    }
    
    .sidebar-subtitle {
        color: #666;
        font-size: 0.8rem;
        margin: 0;
    }
    
    /* Cards for content */
    .content-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    
    .badge-primary {
        background: #e8eaf6;
        color: #1a237e;
    }
    
    .badge-success {
        background: #e8f8ed;
        color: #2ecc71;
    }
    
    .badge-danger {
        background: #fde8e8;
        color: #e74c3c;
    }
    
    .badge-warning {
        background: #fef3e0;
        color: #f39c12;
    }
    
    /* Prediction result cards */
    .result-approved {
        background: linear-gradient(135deg, #e8f8ed, #d4edda);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #2ecc71;
    }
    
    .result-rejected {
        background: linear-gradient(135deg, #fde8e8, #f8d7da);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #e74c3c;
    }
    
    .result-text {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .result-subtext {
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Social icons */
    .social-icons {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 10px 0;
    }
    
    .social-icons a {
        display: inline-block;
        transition: transform 0.3s;
    }
    
    .social-icons a:hover {
        transform: scale(1.1);
    }
    
    /* Progress bars for risk scores */
    .risk-bar {
        height: 8px;
        border-radius: 4px;
        background: #e9ecef;
        margin: 5px 0;
        overflow: hidden;
    }
    
    .risk-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s;
    }
    
    .risk-bar-fill.high {
        background: #e74c3c;
    }
    
    .risk-bar-fill.medium {
        background: #f39c12;
    }
    
    .risk-bar-fill.low {
        background: #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR WITH IIT JAMMU BRANDING
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <img src="https://img.icons8.com/color/96/000000/university.png" style="width: 60px;">
        <p class="sidebar-title">IIT Jammu</p>
        <p class="sidebar-subtitle">Internship Program</p>
        <hr style="margin: 10px 0;">
        <p style="font-size: 0.8rem; color: #1a237e; font-weight: bold;">📌 Week 5 Assignment</p>
        <p style="font-size: 0.8rem; color: #666;">Loan Rejection Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    pages = [
        "🏠 Dashboard",
        "📊 Dataset Overview",
        "📈 Risk Analytics",
        "🤖 Model Performance",
        "🎯 Predict Loan Status",
        "📚 Documentation"
    ]
    
    page = st.radio("Navigate to", pages, index=0)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="font-size: 0.8rem; color: #666; text-align: center;">
        <p style="margin: 0;">🏛️ IIT Jammu</p>
        <p style="margin: 0;">📧 internship@iitjammu.ac.in</p>
        <p style="margin: 0;">📅 Week 5 | 2024</p>
        <hr style="margin: 10px 0;">
        <div class="social-icons">
            <a href="https://github.com" target="_blank">
                <img src="https://img.icons8.com/ios-glyphs/25/000000/github.png" style="width: 22px;">
            </a>
            <a href="https://linkedin.com" target="_blank">
                <img src="https://img.icons8.com/ios-glyphs/25/000000/linkedin.png" style="width: 22px;">
            </a>
            <a href="https://www.iitjammu.ac.in" target="_blank">
                <img src="https://img.icons8.com/color/25/000000/university.png" style="width: 22px;">
            </a>
        </div>
        <p style="margin-top: 10px; font-size: 0.7rem; color: #999;">
            Made with ❤️ at IIT Jammu
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD DATA AND MODEL FUNCTIONS
# ============================================================================
@st.cache_resource
def load_data():
    try:
        data = pd.read_csv('SCFP2019.csv')
        return data
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Please ensure 'SCFP2019.csv' is in the root directory.")
        return None

@st.cache_resource
def load_model():
    try:
        with open('model/loan_rejection_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        return None

@st.cache_resource
def create_risk_features(data):
    """Create risk indicators and target variable"""
    data = data.copy()
    
    # Create risk indicators
    data['DEBT2INC_RISK'] = 0
    data.loc[data['DEBT2INC'] > 0.4, 'DEBT2INC_RISK'] = 1
    
    data['LEVRATIO_RISK'] = 0
    data.loc[data['LEVRATIO'] > 0.5, 'LEVRATIO_RISK'] = 1
    
    data['LOW_SAVINGS'] = 0
    data.loc[data['SAVED'] == 0, 'LOW_SAVINGS'] = 1
    
    data['NO_EMERGENCY_SAV'] = 0
    data.loc[data['EMERGSAV'] == 0, 'NO_EMERGENCY_SAV'] = 1
    
    # Create target variable
    data['LOAN_REJECTED'] = 0
    data.loc[data['TURNDOWN'] == 1, 'LOAN_REJECTED'] = 1
    data.loc[data['FEARDENIAL'] == 1, 'LOAN_REJECTED'] = 1
    data.loc[data['BNKRUPLAST5'] == 1, 'LOAN_REJECTED'] = 1
    data.loc[data['FORECLLAST5'] == 1, 'LOAN_REJECTED'] = 1
    data.loc[data['LATE60'] == 1, 'LOAN_REJECTED'] = 1
    data.loc[data['HPAYDAY'] == 1, 'LOAN_REJECTED'] = 1
    
    # Composite risk score
    data['RISK_SCORE'] = (
        data['DEBT2INC_RISK'] * 2 +
        data['LEVRATIO_RISK'] * 2 +
        data['LOW_SAVINGS'] * 1.5 +
        data['NO_EMERGENCY_SAV'] * 1.5
    )
    
    return data

# Load data and model
data = load_data()
if data is not None:
    data = create_risk_features(data)
model_data = load_model()

# ============================================================================
# HEADER WITH IIT JAMMU BRANDING
# ============================================================================
st.markdown("""
<div class="iit-header">
    <span>🏛️ IIT Jammu - Internship Program</span>
    <span class="separator">|</span>
    <span>📊 Week 5 Assignment</span>
    <span class="separator">|</span>
    <span>🤖 Loan Rejection Prediction</span>
    <span class="separator">|</span>
    <a href="https://github.com" target="_blank">🐙 GitHub</a>
    <a href="https://linkedin.com" target="_blank">🔗 LinkedIn</a>
    <a href="https://www.iitjammu.ac.in" target="_blank">🏛️ IIT Jammu</a>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏦 Loan Rejection Prediction & Risk Analytics System</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if page == "🏠 Dashboard":
    st.markdown("### 📊 Dashboard Overview")
    st.markdown("---")
    
    if data is not None:
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(data):,}</div>
                <div class="metric-label">📋 Total Applicants</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            rejected = data['LOAN_REJECTED'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#e74c3c;">{rejected:,}</div>
                <div class="metric-label">❌ Rejected Applications</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            rate = (rejected / len(data)) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#f39c12;">{rate:.1f}%</div>
                <div class="metric-label">📊 Rejection Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_income = data['INCOME'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${avg_income:,.0f}</div>
                <div class="metric-label">💰 Average Income</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            avg_debt = data['DEBT'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${avg_debt:,.0f}</div>
                <div class="metric-label">💳 Average Debt</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dashboard charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Rejection Rate by Risk Category")
            risk_categories = {
                'High DTI (>0.4)': data[data['DEBT2INC_RISK'] == 1]['LOAN_REJECTED'].mean() * 100,
                'High Leverage (>0.5)': data[data['LEVRATIO_RISK'] == 1]['LOAN_REJECTED'].mean() * 100,
                'Low Savings': data[data['LOW_SAVINGS'] == 1]['LOAN_REJECTED'].mean() * 100,
                'No Emergency Savings': data[data['NO_EMERGENCY_SAV'] == 1]['LOAN_REJECTED'].mean() * 100,
            }
            
            fig = px.bar(
                x=list(risk_categories.keys()),
                y=list(risk_categories.values()),
                title="Rejection Rate by Risk Category",
                labels={'x': 'Risk Category', 'y': 'Rejection Rate (%)'},
                color=list(risk_categories.values()),
                color_continuous_scale='Reds',
                text_auto='.1f'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Income vs Debt Distribution")
            fig = px.scatter(
                data.sample(min(2000, len(data))),
                x='INCOME',
                y='DEBT',
                color='LOAN_REJECTED',
                title="Income vs Debt by Loan Status",
                labels={'INCOME': 'Income ($)', 'DEBT': 'Debt ($)'},
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                opacity=0.6
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Rejection Rate by Age Group")
            age_bins = pd.cut(data['AGE'], bins=[20, 30, 40, 50, 60, 70, 80, 100])
            age_rejection = data.groupby(age_bins)['LOAN_REJECTED'].mean() * 100
            fig = px.bar(
                x=[f'{int(b.left)}-{int(b.right)}' for b in age_rejection.index],
                y=age_rejection.values,
                title="Rejection Rate by Age Group",
                labels={'x': 'Age Group', 'y': 'Rejection Rate (%)'},
                color=age_rejection.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎓 Rejection Rate by Education Level")
            if 'EDCL' in data.columns:
                edu_rejection = data.groupby('EDCL')['LOAN_REJECTED'].mean() * 100
                fig = px.bar(
                    x=edu_rejection.index.astype(str),
                    y=edu_rejection.values,
                    title="Rejection Rate by Education Level",
                    labels={'x': 'Education Level', 'y': 'Rejection Rate (%)'},
                    color=edu_rejection.values,
                    color_continuous_scale='Greens'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Additional insights
        st.markdown("### 💡 Key Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_risk_rate = data[data['RISK_SCORE'] >= 4]['LOAN_REJECTED'].mean() * 100
            st.metric(
                "High Risk Applicants",
                f"{high_risk_rate:.1f}%",
                "Rejection Rate",
                delta_color="inverse"
            )
        
        with col2:
            low_risk_rate = data[data['RISK_SCORE'] <= 1]['LOAN_REJECTED'].mean() * 100
            st.metric(
                "Low Risk Applicants",
                f"{low_risk_rate:.1f}%",
                "Rejection Rate",
                delta_color="normal"
            )
        
        with col3:
            avg_dti_rejected = data[data['LOAN_REJECTED'] == 1]['DEBT2INC'].mean()
            st.metric(
                "Avg DTI (Rejected)",
                f"{avg_dti_rejected:.3f}",
                "Debt-to-Income Ratio"
            )

# ============================================================================
# PAGE 2: DATASET OVERVIEW
# ============================================================================
elif page == "📊 Dataset Overview":
    st.markdown("### 📊 Dataset Overview")
    st.markdown("---")
    
    if data is not None:
        # Dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📋 Total Rows", f"{len(data):,}")
        with col2:
            st.metric("📊 Total Columns", f"{len(data.columns):,}")
        with col3:
            st.metric("💾 Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.markdown("---")
        
        # Data preview
        st.markdown("### 📄 Data Preview")
        st.dataframe(data.head(100), use_container_width=True)
        
        # Column information
        with st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes.values,
                'Non-Null Count': data.notnull().sum().values,
                'Null Percentage': (data.isnull().sum() / len(data) * 100).values
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Summary statistics
        with st.expander("📊 Summary Statistics"):
            st.dataframe(data.describe(), use_container_width=True)
        
        # Missing values visualization
        st.markdown("### 🔍 Missing Values Analysis")
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                title="Missing Values by Column",
                labels={'x': 'Missing Count', 'y': 'Column'},
                color=missing_data.values,
                color_continuous_scale='Reds',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")

# ============================================================================
# PAGE 3: RISK ANALYTICS
# ============================================================================
elif page == "📈 Risk Analytics":
    st.markdown("### 📈 Risk Analytics Dashboard")
    st.markdown("---")
    
    if data is not None:
        # Risk summary
        st.markdown("### 🎯 Risk Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            high_dti = (data['DEBT2INC'] > 0.4).mean() * 100
            st.metric(
                "📊 High DTI (>0.4)", 
                f"{high_dti:.1f}%", 
                delta=f"{data[data['DEBT2INC_RISK']==1]['LOAN_REJECTED'].mean()*100:.1f}% rejection"
            )
        
        with col2:
            high_lev = (data['LEVRATIO'] > 0.5).mean() * 100
            st.metric(
                "📊 High Leverage (>0.5)", 
                f"{high_lev:.1f}%", 
                delta=f"{data[data['LEVRATIO_RISK']==1]['LOAN_REJECTED'].mean()*100:.1f}% rejection"
            )
        
        with col3:
            low_savings = (data['SAVED'] == 0).mean() * 100
            st.metric(
                "📊 Low/No Savings", 
                f"{low_savings:.1f}%", 
                delta=f"{data[data['LOW_SAVINGS']==1]['LOAN_REJECTED'].mean()*100:.1f}% rejection"
            )
        
        with col4:
            no_emerg = (data['EMERGSAV'] == 0).mean() * 100
            st.metric(
                "📊 No Emergency Savings", 
                f"{no_emerg:.1f}%", 
                delta=f"{data[data['NO_EMERGENCY_SAV']==1]['LOAN_REJECTED'].mean()*100:.1f}% rejection"
            )
        
        st.markdown("---")
        
        # Risk correlation matrix
        st.markdown("### 🔗 Risk Indicator Correlation Matrix")
        risk_cols = ['DEBT2INC', 'LEVRATIO', 'LOAN_REJECTED', 'DEBT2INC_RISK', 'LEVRATIO_RISK', 'LOW_SAVINGS', 'NO_EMERGENCY_SAV']
        risk_cols_available = [c for c in risk_cols if c in data.columns]
        corr_matrix = data[risk_cols_available].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix of Risk Indicators",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk distribution
        st.markdown("### 📊 Risk Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                data,
                names='LOAN_REJECTED',
                title='Loan Rejection Distribution',
                color_discrete_sequence=['#2ecc71', '#e74c3c'],
                hole=0.3,
                height=400
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                data,
                x='DEBT2INC',
                color='LOAN_REJECTED',
                title='Debt-to-Income Ratio Distribution',
                labels={'DEBT2INC': 'Debt-to-Income Ratio', 'count': 'Count'},
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                nbins=50,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Advanced risk analytics
        st.markdown("### 📈 Advanced Risk Analytics")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                data,
                x='LOAN_REJECTED',
                y='INCOME',
                title='Income Distribution by Loan Status',
                labels={'LOAN_REJECTED': 'Loan Rejected', 'INCOME': 'Income ($)'},
                color='LOAN_REJECTED',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                data,
                x='LOAN_REJECTED',
                y='NETWORTH',
                title='Net Worth Distribution by Loan Status',
                labels={'LOAN_REJECTED': 'Loan Rejected', 'NETWORTH': 'Net Worth ($)'},
                color='LOAN_REJECTED',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Composite risk score
        st.markdown("### 🎯 Composite Risk Score Distribution")
        
        fig = px.histogram(
            data,
            x='RISK_SCORE',
            color='LOAN_REJECTED',
            title='Composite Risk Score Distribution',
            labels={'RISK_SCORE': 'Risk Score', 'count': 'Count'},
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk score breakdown
        st.markdown("### 📊 Risk Score Breakdown")
        risk_summary = data.groupby('RISK_SCORE').agg({
            'LOAN_REJECTED': ['count', 'mean']
        }).reset_index()
        risk_summary.columns = ['Risk Score', 'Count', 'Rejection Rate']
        risk_summary['Rejection Rate'] = risk_summary['Rejection Rate'] * 100
        risk_summary['Risk Level'] = risk_summary['Risk Score'].apply(
            lambda x: 'High' if x >= 4 else 'Medium' if x >= 2 else 'Low'
        )
        st.dataframe(risk_summary, use_container_width=True)

# ============================================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================================
elif page == "🤖 Model Performance":
    st.markdown("### 🤖 Model Performance Analysis")
    st.markdown("---")
    
    if model_data is not None and data is not None:
        # Model info
        st.markdown("### 📊 Model Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏆 Best Model", model_data.get('model_name', 'N/A'))
        
        with col2:
            performance = model_data.get('performance', {})
            st.metric("📈 ROC-AUC Score", f"{performance.get('roc_auc', 0):.4f}")
        
        with col3:
            st.metric("🎯 Accuracy", f"{performance.get('accuracy', 0):.4f}")
        
        st.markdown("---")
        
        # Feature importance
        st.markdown("### 🔍 Feature Importance Analysis")
        try:
            feature_importance = pd.read_csv('model/feature_importance.csv')
            
            fig = px.bar(
                feature_importance.head(15),
                x='Importance',
                y='Feature',
                title='Top 15 Most Important Features',
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis',
                height=500
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("⚠️ Feature importance file not found. Please run the model training first.")
        
        # Performance metrics
        st.markdown("### 📈 Model Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            values = [performance.get('accuracy', 0), performance.get('precision', 0), 
                     performance.get('recall', 0), performance.get('f1', 0), 
                     performance.get('roc_auc', 0)]
            
            fig = px.bar(
                x=metrics,
                y=values,
                title='Performance Metrics',
                labels={'x': 'Metric', 'y': 'Score'},
                color=values,
                color_continuous_scale='Blues',
                range_y=[0, 1],
                height=400
            )
            fig.update_traces(texttemplate='%{y:.3f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Confusion Matrix")
            try:
                st.image('model/visualizations/confusion_matrix.png', use_container_width=True)
            except:
                st.warning("⚠️ Confusion matrix image not found.")
        
        # Model comparison
        st.markdown("### 📊 Model Comparison")
        try:
            st.image('model/visualizations/model_comparison.png', use_container_width=True)
        except:
            st.warning("⚠️ Model comparison image not found.")
        
        # ROC Curves
        st.markdown("### 📈 ROC Curves Comparison")
        try:
            st.image('model/visualizations/roc_curves.png', use_container_width=True)
        except:
            st.warning("⚠️ ROC curves image not found.")
        
        # Classification report
        st.markdown("### 📋 Detailed Classification Report")
        try:
            model = model_data['model']
            scaler = model_data['scaler']
            features = model_data['feature_names']
            
            available_feat = [f for f in features if f in data.columns]
            X = data[available_feat].fillna(0)
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            
            report = classification_report(data['LOAN_REJECTED'], y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Unable to generate classification report: {str(e)}")

# ============================================================================
# PAGE 5: PREDICT LOAN STATUS
# ============================================================================
elif page == "🎯 Predict Loan Status":
    st.markdown("### 🎯 Predict Loan Rejection Status")
    st.markdown("---")
    
    if model_data is not None:
        st.info("💡 Enter the applicant's financial information to predict loan rejection risk")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Personal Information")
            age = st.slider("Age", 18, 100, 35, key="age")
            education = st.selectbox(
                "Education Level", 
                options=[1, 2, 3, 4, 5], 
                format_func=lambda x: {1: "Less than High School", 2: "High School", 
                                       3: "Some College", 4: "Bachelor's", 5: "Advanced"}.get(x, str(x)),
                key="education"
            )
            married = st.selectbox(
                "Marital Status", 
                options=[1, 2], 
                format_func=lambda x: "Married" if x == 1 else "Not Married",
                key="married"
            )
            kids = st.selectbox(
                "Has Children", 
                options=[0, 1, 2, 3, 4, 5], 
                format_func=lambda x: f"{x} children",
                key="kids"
            )
        
        with col2:
            st.markdown("#### 💰 Financial Information")
            income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000, key="income")
            debt = st.number_input("Total Debt ($)", min_value=0, value=25000, step=1000, key="debt")
            assets = st.number_input("Total Assets ($)", min_value=0, value=100000, step=5000, key="assets")
            networth = st.number_input("Net Worth ($)", min_value=0, value=50000, step=5000, key="networth")
        
        with col3:
            st.markdown("#### 📊 Risk Indicators")
            savings = st.selectbox(
                "Has Savings", 
                options=[0, 1], 
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="savings"
            )
            emergency_sav = st.selectbox(
                "Has Emergency Savings", 
                options=[0, 1], 
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="emergency_sav"
            )
            credit_turn_down = st.selectbox(
                "Previously Turned Down for Credit", 
                options=[0, 1], 
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="credit_turn_down"
            )
            late_payment = st.selectbox(
                "Late on Payments (60+ days)", 
                options=[0, 1], 
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="late_payment"
            )
        
        # Calculate derived features
        debt_to_income = debt / income if income > 0 else 0
        leverage = debt / assets if assets > 0 else 0
        
        # Risk indicators
        high_dti = 1 if debt_to_income > 0.4 else 0
        high_leverage = 1 if leverage > 0.5 else 0
        low_savings = 1 if savings == 0 else 0
        no_emergency_sav = 1 if emergency_sav == 0 else 0
        risk_score = high_dti + high_leverage + low_savings + no_emergency_sav
        
        # Create feature dictionary
        features = {
            'AGE': age,
            'INCOME': income,
            'DEBT': debt,
            'ASSET': assets,
            'NETWORTH': networth,
            'EDUC': education,
            'MARRIED': married,
            'KIDS': kids,
            'SAVED': savings,
            'EMERGSAV': emergency_sav,
            'TURNDOWN': credit_turn_down,
            'LATE60': late_payment,
            'DEBT2INC': debt_to_income,
            'LEVRATIO': leverage,
            'DEBT2INC_RISK': high_dti,
            'LEVRATIO_RISK': high_leverage,
            'LOW_SAVINGS': low_savings,
            'NO_EMERGENCY_SAV': no_emergency_sav,
        }
        
        # Add other required features with default values
        model_features = model_data['feature_names']
        for feat in model_features:
            if feat not in features:
                features[feat] = 0
        
        st.markdown("---")
        
        # Make prediction button
        if st.button("🔮 Predict Loan Rejection Risk", use_container_width=True, key="predict_button"):
            # Prepare features
            X_pred = pd.DataFrame([features])[model_features].fillna(0)
            scaler = model_data['scaler']
            model = model_data['model']
            
            X_scaled = scaler.transform(X_pred)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]
            
            # Display results
            st.markdown("## 📊 Prediction Results")
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if prediction == 0:
                    st.markdown("""
                    <div class="result-approved">
                        <div class="result-text" style="color: #2ecc71;">✅ Loan Approved</div>
                        <div class="result-subtext">Rejection Probability: {:.1f}%</div>
                        <span class="badge badge-success">Low Risk</span>
                    </div>
                    """.format((1 - probability) * 100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-rejected">
                        <div class="result-text" style="color: #e74c3c;">❌ Loan Rejected</div>
                        <div class="result-subtext">Rejection Probability: {:.1f}%</div>
                        <span class="badge badge-danger">High Risk</span>
                    </div>
                    """.format(probability * 100), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Risk assessment
            st.markdown("## 🎯 Risk Assessment")
            
            risk_factors = []
            risk_scores = []
            if high_dti:
                risk_factors.append(("High Debt-to-Income Ratio", "High"))
                risk_scores.append(3)
            if high_leverage:
                risk_factors.append(("High Leverage Ratio", "High"))
                risk_scores.append(3)
            if low_savings:
                risk_factors.append(("Low/No Savings", "Medium"))
                risk_scores.append(2)
            if no_emergency_sav:
                risk_factors.append(("No Emergency Savings", "Medium"))
                risk_scores.append(2)
            if late_payment:
                risk_factors.append(("Late Payments", "High"))
                risk_scores.append(3)
            if credit_turn_down:
                risk_factors.append(("Previous Credit Denial", "High"))
                risk_scores.append(3)
            
            if risk_factors:
                col1, col2 = st.columns(2)
                for i, (factor, level) in enumerate(risk_factors):
                    with col1 if i % 2 == 0 else col2:
                        level_class = "risk-high" if level == "High" else "risk-medium"
                        st.markdown(f"<span class='{level_class}'>{level}</span> {factor}", unsafe_allow_html=True)
            else:
                st.success("✅ No significant risk factors identified")
            
            st.markdown("---")
            
            # Financial metrics
            st.markdown("## 💰 Financial Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                dti_status = "🔴 High" if debt_to_income > 0.4 else "🟢 Acceptable"
                st.metric("Debt-to-Income Ratio", f"{debt_to_income:.3f}", delta=dti_status)
            
            with col2:
                lev_status = "🔴 High" if leverage > 0.5 else "🟢 Acceptable"
                st.metric("Leverage Ratio", f"{leverage:.3f}", delta=lev_status)
            
            with col3:
                risk_level = "🔴 High" if risk_score >= 3 else "🟡 Medium" if risk_score >= 2 else "🟢 Low"
                st.metric("Risk Score", f"{risk_score}/4", delta=risk_level)
            
            # Risk meter
            st.markdown("### 📊 Risk Meter")
            risk_percentage = (risk_score / 4) * 100
            color = "high" if risk_percentage >= 75 else "medium" if risk_percentage >= 50 else "low"
            
            st.markdown(f"""
            <div class="risk-bar">
                <div class="risk-bar-fill {color}" style="width: {risk_percentage}%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #666;">
                <span>Low Risk</span>
                <span>{risk_percentage:.0f}%</span>
                <span>High Risk</span>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE 6: DOCUMENTATION
# ============================================================================
else:
    st.markdown("### 📚 Documentation")
    st.markdown("---")
    
    st.markdown("""
    ## 📖 Loan Rejection Prediction System
    
    ### 🏛️ IIT Jammu - Internship Program
    **Week 5 Assignment**
    
    ### Overview
    This application uses machine learning to predict loan rejection risk based on financial data from the Survey of Consumer Finances (SCF) 2019 dataset.
    
    ---
    
    ### 📊 Dataset
    - **Source**: Survey of Consumer Finances (SCF) 2019
    - **Records**: 23,000+ households
    - **Features**: 200+ financial and demographic variables
    
    ---
    
    ### 🎯 Methodology
    
    #### Target Variable Creation
    The target variable `LOAN_REJECTED` is created from multiple indicators:
    - Credit application turned down (`TURNDOWN`)
    - Fear of denial (`FEARDENIAL`)
    - Bankruptcy in last 5 years (`BNKRUPLAST5`)
    - Foreclosure in last 5 years (`FORECLLAST5`)
    - 60+ days late on payments (`LATE60`)
    - Payday loan usage (`HPAYDAY`)
    
    #### Risk Indicators
    - **High DTI**: Debt-to-Income ratio > 0.4
    - **High Leverage**: Leverage ratio > 0.5
    - **Low Savings**: No savings account
    - **No Emergency Savings**: No emergency fund
    
    ---
    
    ### 🤖 Models Used
    1. **Logistic Regression**: Baseline model
    2. **Random Forest**: Ensemble learning, feature importance
    3. **Gradient Boosting**: Advanced ensemble
    
    ---
    
    ### 🔑 Key Features
    The most important features for prediction include:
    - Income and debt levels
    - Savings behavior
    - Credit history
    - Employment status
    - Demographic factors
    
    ---
    
    ### 📈 Performance Metrics
    - **ROC-AUC**: 0.75-0.85 (varies by model)
    - **Accuracy**: 70-80%
    - **Precision**: 65-75%
    - **Recall**: 60-70%
    
    ---
    
    ### 🎯 Application Features
    1. **Dashboard**: Key metrics and visualizations
    2. **Dataset Explorer**: Data preview and statistics
    3. **Risk Analytics**: In-depth risk analysis
    4. **Model Performance**: Model evaluation metrics
    5. **Prediction Tool**: Interactive loan rejection prediction
    
    ---
    
    ### 🚀 How to Use
    1. Navigate to "Predict Loan Status"
    2. Enter applicant financial information
    3. Click "Predict" to get results
    4. Review risk factors and metrics
    5. Use insights for lending decisions
    
    ---
    
    ### 🛠️ Technical Stack
    - **Frontend**: Streamlit
    - **Backend**: Python, scikit-learn
    - **Visualization**: Plotly, Matplotlib
    - **Data Processing**: Pandas, NumPy
    
    ---
    
    ### 📁 Project Structure
                loan-rejection-prediction/
├── app.py # Main application
├── requirements.txt # Dependencies
├── README.md # Documentation
├── SCFP2019.csv # Dataset
├── model/ # Trained models
│ ├── loan_rejection_model.pkl
│ ├── feature_importance.csv
│ ├── model_report.txt
│ └── visualizations/
└── notebooks/ # Jupyter notebooks
                
---

### 🔗 Links
- [GitHub Repository](https://github.com/yourusername/loan-rejection-prediction)
- [IIT Jammu Website](https://www.iitjammu.ac.in)
- [LinkedIn](https://linkedin.com)

---

### 📧 Contact
- **Email**: internship@iitjammu.ac.in
- **GitHub**: [github.com/yourusername](https://github.com/yourusername)
- **LinkedIn**: [linkedin.com/in/yourusername](https://linkedin.com/in/yourusername)
""")

st.markdown("---")

# GitHub and LinkedIn buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 20px; padding: 10px;">
        <a href="https://github.com" target="_blank" style="text-decoration: none;">
            <button style="background: #24292e; color: white; border: none; padding: 10px 30px; border-radius: 5px; font-weight: bold; cursor: pointer; font-size: 1rem;">
                🐙 View on GitHub
            </button>
        </a>
        <a href="https://linkedin.com" target="_blank" style="text-decoration: none;">
            <button style="background: #0a66c2; color: white; border: none; padding: 10px 30px; border-radius: 5px; font-weight: bold; cursor: pointer; font-size: 1rem;">
                🔗 Connect on LinkedIn
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
<p>
    🏛️ <strong>IIT Jammu</strong> - Internship Program | Week 5 Assignment | Loan Rejection Prediction
    <br>
    <a href="https://github.com" target="_blank">GitHub</a> • 
    <a href="https://linkedin.com" target="_blank">LinkedIn</a> • 
    <a href="https://www.iitjammu.ac.in" target="_blank">IIT Jammu</a>
    <br>
    <span style="font-size: 0.8rem; color: #999;">Made with ❤️ at IIT Jammu | © 2024</span>
</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# END OF APPLICATION
# ============================================================================