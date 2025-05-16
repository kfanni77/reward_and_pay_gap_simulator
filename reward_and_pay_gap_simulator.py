# reward_and_pay_gap_simulator.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import statsmodels.api as sm

# --- Load Data ---
df = pd.read_excel("Dummy_HR_Compensation_Dataset.xlsx")
df['BaseSalary_Original'] = df['BaseSalary']

# --- Salary Bands ---
salary_band_summary = df.groupby('Level')['BaseSalary'].agg(['min', 'median', 'max']).reset_index()
salary_band_summary.columns = ['Level', 'BandMin', 'BandMid', 'BandMax']
df = df.merge(salary_band_summary, on='Level', how='left')
df['BandPosition'] = (df['BaseSalary'] - df['BandMin']) / (df['BandMax'] - df['BandMin'])
df['BandPosition'] = df['BandPosition'].clip(0, 1.5)

# --- Sidebar Controls ---
st.sidebar.title("Base Merit % by Performance Rating")
base_merit_percent = {
    1: st.sidebar.slider("Rating 1", 0.00, 0.05, 0.00, step=0.005),
    2: st.sidebar.slider("Rating 2", 0.00, 0.05, 0.01, step=0.005),
    3: st.sidebar.slider("Rating 3", 0.00, 0.05, 0.02, step=0.005),
    4: st.sidebar.slider("Rating 4", 0.00, 0.08, 0.03, step=0.005),
    5: st.sidebar.slider("Rating 5", 0.00, 0.12, 0.05, step=0.005),
}

df['BaseMeritPct'] = df['PerformanceRating'].map(base_merit_percent)

def adjustment_factor(pos):
    if pos < 0.0: return 1.5
    elif pos < 0.5: return 1.2
    elif pos <= 1.0: return 1.0
    elif pos <= 1.2: return 0.5
    else: return 0.0

df['BandAdjustment'] = df['BandPosition'].apply(adjustment_factor)
df['AdjustedMeritPct'] = df['BaseMeritPct'] * df['BandAdjustment']
df['FinalMeritIncrease'] = df['BaseSalary'] * df['AdjustedMeritPct']
df['FinalMeritPct'] = df['AdjustedMeritPct']
df['BaseSalary'] = df['BaseSalary_Original'] + df['FinalMeritIncrease']

# --- Bonus Logic ---
df['TeamScore'] = np.random.randint(1, 6, size=len(df))
df['RetentionRisk'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
df['NormPerf'] = df['PerformanceRating'] / df['PerformanceRating'].max()
df['NormTeam'] = df['TeamScore'] / df['TeamScore'].max()
df['NormRisk'] = df['RetentionRisk']

st.sidebar.title("Bonus Allocation Weights")
w_perf = st.sidebar.slider("Performance Weight", 0.0, 1.0, 0.5, step=0.05)
w_team = st.sidebar.slider("Team Score Weight", 0.0, 1.0, 0.3, step=0.05)
w_ret = st.sidebar.slider("Retention Risk Weight", 0.0, 1.0, 0.2, step=0.05)

total_weight = w_perf + w_team + w_ret
w_perf /= total_weight
w_team /= total_weight
w_ret /= total_weight

bonus_mean = df['Bonus'].mean()
df['SystemBonusRecommendation'] = (
    w_perf * df['NormPerf'] + w_team * df['NormTeam'] + w_ret * df['NormRisk']
) * bonus_mean * 2
df['BonusDelta_vs_System'] = df['Bonus'] - df['SystemBonusRecommendation']

# --- Gender Pay Gap Calculations ---
st.subheader("Gender Pay Gap Impact")
avg_salary_gender_before = df.groupby('Gender')['BaseSalary_Original'].mean()
avg_salary_gender_after = df.groupby('Gender')['BaseSalary'].mean()

unadj_gpg_before = ((avg_salary_gender_before['Male'] - avg_salary_gender_before['Female']) / avg_salary_gender_before['Male']) * 100
unadj_gpg_after = ((avg_salary_gender_after['Male'] - avg_salary_gender_after['Female']) / avg_salary_gender_after['Male']) * 100
unadj_gpg_delta = unadj_gpg_after - unadj_gpg_before
unadj_delta_color = "normal" if abs(unadj_gpg_after) > abs(unadj_gpg_before) else "inverse"

st.metric("Unadjusted GPG (Before, %)", f"{unadj_gpg_before:.2f}%")
st.metric("Unadjusted GPG (After, %)", f"{unadj_gpg_after:.2f}%", delta=f"{unadj_gpg_delta:+.2f}%", delta_color=unadj_delta_color)

# --- Adjusted GPG OLS ---
df_encoded = pd.get_dummies(df.copy(), columns=['Gender', 'Level', 'Department'], drop_first=True)
reg_columns = ['TenureYears', 'PerformanceRating'] + [col for col in df_encoded.columns if col.startswith(('Level_', 'Department_', 'Gender_'))]

X = df_encoded[reg_columns].astype(float)
X = sm.add_constant(X)
y_before = pd.to_numeric(df_encoded['BaseSalary_Original'], errors='coerce')
y_after = pd.to_numeric(df_encoded['BaseSalary'], errors='coerce')

Xb = pd.concat([X, y_before], axis=1).dropna()
Xa = pd.concat([X, y_after], axis=1).dropna()

X_before = Xb[X.columns]
y_before = Xb[y_before.name]
X_after = Xa[X.columns]
y_after = Xa[y_after.name]

model_before = sm.OLS(y_before, X_before).fit()
model_after = sm.OLS(y_after, X_after).fit()

adj_gap_before = model_before.params.get('Gender_Male', 0)
adj_gap_after = model_after.params.get('Gender_Male', 0)
adj_gap_delta = adj_gap_after - adj_gap_before

f_avg_before = avg_salary_gender_before['Female']
f_avg_after = avg_salary_gender_after['Female']
adj_gpg_before_pct = (adj_gap_before / f_avg_before) * 100
adj_gpg_after_pct = (adj_gap_after / f_avg_after) * 100
adj_gpg_delta_pct = adj_gpg_after_pct - adj_gpg_before_pct

color_eur = "normal" if adj_gap_after < adj_gap_before else "inverse"
color_pct = "normal" if adj_gpg_after_pct < adj_gpg_before_pct else "inverse"

st.metric("Adjusted GPG (Before, EUR)", f"€{adj_gap_before:.2f}")
st.metric("Adjusted GPG (After, EUR)", f"€{adj_gap_after:.2f}", delta=f"€{adj_gap_delta:+.2f}", delta_color=color_eur)

st.metric("Adjusted GPG (Before, %)", f"{adj_gpg_before_pct:.2f}%")
st.metric("Adjusted GPG (After, %)", f"{adj_gpg_after_pct:.2f}%", delta=f"{adj_gpg_delta_pct:+.2f}%", delta_color=color_pct)

# --- Download ---
def convert_df(df):
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output

st.download_button(
    label="📥 Download Merit & Bonus Table",
    data=convert_df(df),
    file_name="Merit_Bonus_Simulation.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
