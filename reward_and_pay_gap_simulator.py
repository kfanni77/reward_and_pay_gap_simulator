import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import statsmodels.api as sm

# --- Simulated Data Load ---
df = pd.read_excel("Dummy_HR_Compensation_Dataset.xlsx")
df['BaseSalary_Original'] = df['BaseSalary']

# Simulate salary bands by level
salary_band_summary = df.groupby('Level')['BaseSalary'].agg(['min', 'median', 'max']).reset_index()
salary_band_summary.columns = ['Level', 'BandMin', 'BandMid', 'BandMax']
df = df.merge(salary_band_summary, on='Level', how='left')

# Calculate band position (0 = min, 0.5 = mid, 1 = max)
df['BandPosition'] = (df['BaseSalary'] - df['BandMin']) / (df['BandMax'] - df['BandMin'])
df['BandPosition'] = df['BandPosition'].clip(0, 1.5)

# User-defined base merit % by performance
st.sidebar.title("Base Merit % by Performance Rating")
base_merit_percent = {
    1: st.sidebar.slider("Rating 1", 0.00, 0.05, 0.00, step=0.005),
    2: st.sidebar.slider("Rating 2", 0.00, 0.05, 0.01, step=0.005),
    3: st.sidebar.slider("Rating 3", 0.00, 0.05, 0.02, step=0.005),
    4: st.sidebar.slider("Rating 4", 0.00, 0.08, 0.03, step=0.005),
    5: st.sidebar.slider("Rating 5", 0.00, 0.12, 0.05, step=0.005),
}
df['BaseMeritPct'] = df['PerformanceRating'].map(base_merit_percent)

# Adjustment factor by salary band position
def adjustment_factor(pos):
    if pos < 0.0:
        return 1.5
    elif pos < 0.5:
        return 1.2
    elif pos <= 1.0:
        return 1.0
    elif pos <= 1.2:
        return 0.5
    else:
        return 0.0

df['BandAdjustment'] = df['BandPosition'].apply(adjustment_factor)
df['AdjustedMeritPct'] = df['BaseMeritPct'] * df['BandAdjustment']
df['RecommendedMeritIncrease'] = df['BaseSalary'] * df['AdjustedMeritPct']

# Sidebar: set total merit budget
MERIT_BUDGET = st.sidebar.number_input("Total Merit Budget (€)", min_value=10000, max_value=1000000, value=200000, step=10000)

# Bonus slider weights
st.sidebar.title("Bonus Allocation Weights")
w_perf = st.sidebar.slider("Performance Weight", 0.0, 1.0, 0.5, step=0.05)
w_team = st.sidebar.slider("Team Score Weight", 0.0, 1.0, 0.3, step=0.05)
w_ret = st.sidebar.slider("Retention Risk Weight", 0.0, 1.0, 0.2, step=0.05)

# Normalize weights
total_weight = w_perf + w_team + w_ret
w_perf /= total_weight
w_team /= total_weight
w_ret /= total_weight

# Filters
st.sidebar.title("Custom Grouping Filters")
selected_level = st.sidebar.selectbox("Select Job Level", options=["All"] + sorted(df['Level'].unique().tolist()))
selected_dept = st.sidebar.selectbox("Select Department", options=["All"] + sorted(df['Department'].unique().tolist()))

filtered_df = df.copy()
if selected_level != "All":
    filtered_df = filtered_df[filtered_df['Level'] == selected_level]
if selected_dept != "All":
    filtered_df = filtered_df[filtered_df['Department'] == selected_dept]

# No scaling — use values based on sliders directly
filtered_df['FinalMeritIncrease'] = filtered_df['BaseSalary'] * filtered_df['AdjustedMeritPct']
filtered_df['FinalMeritPct'] = filtered_df['AdjustedMeritPct']
filtered_df['BaseSalary'] = filtered_df['BaseSalary_Original'] + filtered_df['FinalMeritIncrease']

# Bonus logic
filtered_df['TeamScore'] = np.random.randint(1, 6, size=len(filtered_df))
filtered_df['RetentionRisk'] = np.random.choice([0, 1], size=len(filtered_df), p=[0.7, 0.3])
filtered_df['NormPerf'] = filtered_df['PerformanceRating'] / filtered_df['PerformanceRating'].max()
filtered_df['NormTeam'] = filtered_df['TeamScore'] / filtered_df['TeamScore'].max()
filtered_df['NormRisk'] = filtered_df['RetentionRisk']

# Bonus recommendation
bonus_mean = filtered_df['Bonus'].mean()
filtered_df['SystemBonusRecommendation'] = (
    w_perf * filtered_df['NormPerf'] + w_team * filtered_df['NormTeam'] + w_ret * filtered_df['NormRisk']
) * bonus_mean * 2
filtered_df['BonusDelta_vs_System'] = filtered_df['Bonus'] - filtered_df['SystemBonusRecommendation']

# --- Streamlit Layout ---
st.title("Merit & Bonus Simulation App")

col1, col2 = st.columns(2)
col1.metric("Merit Budget", f"€{MERIT_BUDGET:,.0f}")
col2.metric("Simulated Merit Spend", f"€{filtered_df['RecommendedMeritIncrease'].sum():,.0f}", delta=f"€{filtered_df['RecommendedMeritIncrease'].sum() - MERIT_BUDGET:,.0f}")

st.subheader("Average Scaled Merit % by Performance Rating")
avg_pct_by_rating = filtered_df.groupby('PerformanceRating')['AdjustedMeritPct'].mean().round(4).reset_index()
st.dataframe(avg_pct_by_rating)

st.subheader("Bonus Allocation Comparison")
col3, col4 = st.columns(2)
col3.metric("Actual Bonus Total", f"€{filtered_df['Bonus'].sum():,.0f}")
col4.metric("System-Recommended Bonus Total", f"€{filtered_df['SystemBonusRecommendation'].sum():,.0f}", delta=f"€{filtered_df['Bonus'].sum() - filtered_df['SystemBonusRecommendation'].sum():,.0f}")

st.subheader("Gender Representation in Filtered Group")
gender_counts = filtered_df['Gender'].value_counts(normalize=True) * 100
fig_gender, ax_gender = plt.subplots()
ax_gender.bar(gender_counts.index, gender_counts.values, color=['steelblue', 'orchid'])
ax_gender.set_ylabel("Percentage")
ax_gender.set_title("Gender Distribution")
for i, v in enumerate(gender_counts.values):
    ax_gender.text(i, v + 1, f"{v:.1f}%", ha='center')
st.pyplot(fig_gender)

# Charts for Pay Gaps
st.subheader("Pay Gap Comparison (Before vs. After)")
# Calculate gaps
male_before = df[df['Gender'] == 'Male']['BaseSalary_Original'].mean()
female_before = df[df['Gender'] == 'Female']['BaseSalary_Original'].mean()
male_after = filtered_df[filtered_df['Gender'] == 'Male']['BaseSalary'].mean()
female_after = filtered_df[filtered_df['Gender'] == 'Female']['BaseSalary'].mean()

unadj_gpg_before = ((male_before - female_before) / male_before) * 100
unadj_gpg_after = ((male_after - female_after) / male_after) * 100

fig_gap, ax_gap = plt.subplots(figsize=(6, 4))
bar_labels = ['Before', 'After']
bar_values = [unadj_gpg_before, unadj_gpg_after]
colors = ['darkred' if v > 0 else 'darkgreen' for v in bar_values]
ax_gap.bar(bar_labels, bar_values, color=colors)
ax_gap.set_ylabel("Unadjusted GPG (%)")
ax_gap.set_title("Unadjusted Pay Gap: Before vs. After")
for i, v in enumerate(bar_values):
    ax_gap.text(i, v + 0.5 if v >= 0 else v - 2, f"{v:.2f}%", ha='center', color='black')
st.pyplot(fig_gap)

# Extended download
company_median = df['BaseSalary_Original'].median()
for gender in ['Male', 'Female']:
    df[f'Diff_CompanyMedian_{gender}'] = df['BaseSalary_Original'] - df[df['Gender'] == gender]['BaseSalary_Original'].median()
    df[f'Diff_DeptMedian_{gender}'] = df.groupby(['Department'])['BaseSalary_Original'].transform(lambda x: x - x[df[df['Gender'] == gender].index].median())
    df[f'Diff_LevelMedian_{gender}'] = df.groupby(['Level'])['BaseSalary_Original'].transform(lambda x: x - x[df[df['Gender'] == gender].index].median())

# Download
st.subheader("Download Simulation Table")
def convert_df(df):
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output

st.download_button(
    label="📥 Download Merit & Bonus Table",
    data=convert_df(filtered_df),
    file_name="Merit_Bonus_Simulation.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
