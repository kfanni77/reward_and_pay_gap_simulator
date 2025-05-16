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

# --- Filters ---
st.sidebar.title("Filters")
selected_levels = st.sidebar.multiselect("Filter by Level", options=sorted(df['Level'].unique()), default=sorted(df['Level'].unique()))
selected_departments = st.sidebar.multiselect("Filter by Department", options=sorted(df['Department'].unique()), default=sorted(df['Department'].unique()))

filtered_df = df[df['Level'].isin(selected_levels) & df['Department'].isin(selected_departments)].copy()

# --- Gender Representation Chart ---
st.sidebar.subheader("Gender Representation")
gender_counts = filtered_df['Gender'].value_counts(normalize=True).mul(100).round(1)
st.sidebar.bar_chart(gender_counts)

# --- Salary Bands ---
salary_band_summary = df.groupby('Level')['BaseSalary'].agg(['min', 'median', 'max']).reset_index()
salary_band_summary.columns = ['Level', 'BandMin', 'BandMid', 'BandMax']
filtered_df = filtered_df.merge(salary_band_summary, on='Level', how='left')

filtered_df['BandPosition'] = (filtered_df['BaseSalary'] - filtered_df['BandMin']) / (filtered_df['BandMax'] - filtered_df['BandMin'])
filtered_df['BandPosition'] = filtered_df['BandPosition'].clip(0, 1.5)

# --- Merit Simulation ---
# Merit Budget
MERIT_BUDGET = st.sidebar.number_input("Total Merit Budget (€)", min_value=10000, max_value=1000000, value=200000, step=10000)

st.sidebar.title("Base Merit % by Performance Rating")
base_merit_percent = {
    1: st.sidebar.slider("Rating 1", 0.00, 0.05, 0.00, step=0.005),
    2: st.sidebar.slider("Rating 2", 0.00, 0.05, 0.01, step=0.005),
    3: st.sidebar.slider("Rating 3", 0.00, 0.05, 0.02, step=0.005),
    4: st.sidebar.slider("Rating 4", 0.00, 0.08, 0.03, step=0.005),
    5: st.sidebar.slider("Rating 5", 0.00, 0.12, 0.05, step=0.005),
}

filtered_df['BaseMeritPct'] = filtered_df['PerformanceRating'].map(base_merit_percent)

def adjustment_factor(pos):
    if pos < 0.0: return 1.5
    elif pos < 0.5: return 1.2
    elif pos <= 1.0: return 1.0
    elif pos <= 1.2: return 0.5
    else: return 0.0

filtered_df['BandAdjustment'] = filtered_df['BandPosition'].apply(adjustment_factor)
filtered_df['AdjustedMeritPct'] = filtered_df['BaseMeritPct'] * filtered_df['BandAdjustment']
filtered_df['FinalMeritIncrease'] = filtered_df['BaseSalary'] * filtered_df['AdjustedMeritPct']
filtered_df['FinalMeritPct'] = filtered_df['AdjustedMeritPct']
filtered_df['BaseSalary'] = filtered_df['BaseSalary_Original'] + filtered_df['FinalMeritIncrease']

# --- Bonus Calculation ---
st.sidebar.title("Bonus Allocation Weights")
w_perf = st.sidebar.slider("Performance Weight", 0.0, 1.0, 0.5, step=0.05)
w_team = st.sidebar.slider("Team Score Weight", 0.0, 1.0, 0.3, step=0.05)
w_ret = st.sidebar.slider("Retention Risk Weight", 0.0, 1.0, 0.2, step=0.05)

total_weight = w_perf + w_team + w_ret
w_perf /= total_weight
w_team /= total_weight
w_ret /= total_weight

filtered_df['TeamScore'] = np.random.randint(1, 6, size=len(filtered_df))
filtered_df['RetentionRisk'] = np.random.choice([0, 1], size=len(filtered_df), p=[0.7, 0.3])
filtered_df['NormPerf'] = filtered_df['PerformanceRating'] / filtered_df['PerformanceRating'].max()
filtered_df['NormTeam'] = filtered_df['TeamScore'] / filtered_df['TeamScore'].max()
filtered_df['NormRisk'] = filtered_df['RetentionRisk']

bonus_mean = filtered_df['Bonus'].mean()
filtered_df['SystemBonusRecommendation'] = (
    w_perf * filtered_df['NormPerf'] + w_team * filtered_df['NormTeam'] + w_ret * filtered_df['NormRisk']
) * bonus_mean * 2
filtered_df['BonusDelta_vs_System'] = filtered_df['Bonus'] - filtered_df['SystemBonusRecommendation']

# --- GPG ---
st.title("Merit & Bonus Simulation App")

col1, col2 = st.columns(2)
col1.metric("Merit Budget", f"€{MERIT_BUDGET:,.0f}")
col2.metric("Simulated Merit Spend", f"€{filtered_df['RecommendedMeritIncrease'].sum():,.0f}",
            delta=f"€{filtered_df['RecommendedMeritIncrease'].sum() - MERIT_BUDGET:,.0f}")

st.subheader("Average Scaled Merit % by Performance Rating")
avg_pct_by_rating = filtered_df.groupby('PerformanceRating')['AdjustedMeritPct'].mean().round(4).reset_index()
st.dataframe(avg_pct_by_rating)

st.subheader("Bonus Allocation Comparison")
col3, col4 = st.columns(2)
col3.metric("Actual Bonus Total", f"€{filtered_df['Bonus'].sum():,.0f}")
col4.metric("System-Recommended Bonus Total", f"€{filtered_df['SystemBonusRecommendation'].sum():,.0f}",
            delta=f"€{filtered_df['Bonus'].sum() - filtered_df['SystemBonusRecommendation'].sum():,.0f}")

st.subheader("Bonus Distribution: Actual vs. Recommended")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=filtered_df, x='SystemBonusRecommendation', y='Bonus', hue='RetentionRisk', palette='coolwarm', alpha=0.6, ax=ax1)
ax1.plot([filtered_df['SystemBonusRecommendation'].min(), filtered_df['SystemBonusRecommendation'].max()],
         [filtered_df['SystemBonusRecommendation'].min(), filtered_df['SystemBonusRecommendation'].max()],
         linestyle='--', color='gray')
ax1.set_title("Actual vs. Recommended Bonus")
ax1.set_xlabel("System-Recommended Bonus (€)")
ax1.set_ylabel("Actual Bonus (€)")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.histplot(filtered_df['BonusDelta_vs_System'], bins=30, kde=True, color='slateblue', ax=ax2)
ax2.axvline(0, linestyle='--', color='black')
ax2.set_title("Bonus Delta: Actual - System Recommendation")
ax2.set_xlabel("Bonus Delta (€)")
ax2.set_ylabel("Employee Count")
st.pyplot(fig2)

st.subheader("Gender Pay Gap Impact")

def calc_unadjusted_gpg(data):
    avg_salary = data.groupby('Gender')['BaseSalary'].mean()
    return ((avg_salary['Male'] - avg_salary['Female']) / avg_salary['Male']) * 100

def calc_adjusted_gpg(dataframe):
    df_encoded = pd.get_dummies(dataframe.copy(), columns=['Gender', 'Level', 'Department'], drop_first=True)
    reg_columns = ['TenureYears', 'PerformanceRating'] + [
        col for col in df_encoded.columns if col.startswith('Level_') or col.startswith('Department_') or col.startswith('Gender_')
    ]
    X = df_encoded[reg_columns].astype(float)
    X = sm.add_constant(X)
    y = pd.to_numeric(df_encoded['BaseSalary'], errors='coerce')
    model = sm.OLS(y, X).fit()
    gap_eur = model.params.get('Gender_Male', np.nan)
    avg_female_salary = dataframe[dataframe['Gender'] == 'Female']['BaseSalary'].mean()
    gap_pct = (gap_eur / avg_female_salary) * 100 if avg_female_salary else np.nan
    return gap_eur, gap_pct

before_unadj = calc_unadjusted_gpg(filtered_df[['Gender', 'BaseSalary_Original']].assign(BaseSalary=filtered_df['BaseSalary_Original']))
after_unadj = calc_unadjusted_gpg(filtered_df)

before_adj_eur, before_adj_pct = calc_adjusted_gpg(filtered_df.assign(BaseSalary=filtered_df['BaseSalary_Original']))
after_adj_eur, after_adj_pct = calc_adjusted_gpg(filtered_df)

col_a, col_b = st.columns(2)
col_a.metric("Unadjusted GPG Before", f"{before_unadj:.2f}%")
col_b.metric("Unadjusted GPG After", f"{after_unadj:.2f}%", delta=f"{after_unadj - before_unadj:+.2f}%", delta_color="inverse" if abs(after_unadj) < abs(before_unadj) else "normal")

col_c, col_d = st.columns(2)
col_c.metric("Adjusted GPG Before (EUR)", f"€{before_adj_eur:.2f}")
col_d.metric("Adjusted GPG After (EUR)", f"€{after_adj_eur:.2f}", delta=f"€{after_adj_eur - before_adj_eur:+.2f}", delta_color="normal" if after_adj_eur < before_adj_eur else "inverse")

col_e, col_f = st.columns(2)
col_e.metric("Adjusted GPG Before (%)", f"{before_adj_pct:.2f}%")
col_f.metric("Adjusted GPG After (%)", f"{after_adj_pct:.2f}%", delta=f"{after_adj_pct - before_adj_pct:+.2f}%", delta_color="normal" if after_adj_pct < before_adj_pct else "inverse")

# --- Charts ---
st.subheader("Pay Gap Visualizations")
fig_gap, ax_gap = plt.subplots(1, 2, figsize=(10, 4))
ax_gap[0].bar(['Before', 'After'], [before_unadj, after_unadj], color=['gray', 'skyblue'])
ax_gap[0].set_title("Unadjusted GPG (%)")
ax_gap[1].bar(['Before', 'After'], [before_adj_eur, after_adj_eur], color=['gray', 'lightgreen'])
ax_gap[1].set_title("Adjusted GPG (EUR)")
st.pyplot(fig_gap)

# --- Extended Download Columns ---
for gender in ['Male', 'Female']:
    df[f'Diff_CompanyMedian_{gender}'] = df['BaseSalary_Original'] - df[df['Gender'] == gender]['BaseSalary_Original'].median()
    df[f'Diff_DeptMedian_{gender}'] = df.groupby('Department')['BaseSalary_Original'].transform(lambda x: x - x[df['Gender'] == gender].median())
    df[f'Diff_LevelMedian_{gender}'] = df.groupby('Level')['BaseSalary_Original'].transform(lambda x: x - x[df['Gender'] == gender].median())

# --- Data Table ---
if st.checkbox("Show Filtered Employee Data"):
    st.dataframe(filtered_df)

# --- Download ---
def convert_df(data):
    output = BytesIO()
    data.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output

st.download_button(
    label="📥 Download Merit & Bonus Table",
    data=convert_df(df),
    file_name="Merit_Bonus_Simulation.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
