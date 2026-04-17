"""
Kerri Riley
Program Evaluation 
"""

# Import needed modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set default plot style for visuals
sns.set(style="whitegrid", palette="Set2")


# Load data and read turnout.csv file
df = pd.read_csv("turnout.csv")


# Convert turnout to percent  (From proportion to percent)
df["turnout_pct"] = df["turnout"] * 100

# Calculates proportion of missing values for each variable
missing = df.isnull().mean().sort_values(ascending=False)


# Codebook function 
def codebook(df):
    print("/nCodebook Summary/n")
    
    for col in df.columns:
        print("\n")
        print(f"Variable: {col}")
        print("\n")

        print(f"Type: {df[col].dtype}")
        print(f"Observations: {len(df)}")
        print(f"Missing: {df[col].isna().sum()} ({df[col].isna().mean():.2%})")
        print(f"Unique values: {df[col].nunique()}")

        if pd.api.types.is_numeric_dtype(df[col]):
            print("\nSummary statistics:")
            desc = df[col].describe(percentiles=[.1, .25, .5, .75, .9])
            print(desc)
        else:
            print("\nTop categories:")
            print(df[col].value_counts().head(10))

# codebook(df) 


# Descriptive Statistics
vars_table1 = [
    "turnout_pct",
    "log_pop",
    "log_inc",
    "per_HSeducation",
    "per_AfricanAmerican",
    "per_urban"
]

table1 = df[vars_table1].describe().T
table1 = table1[["count", "mean", "std", "min", "max"]]

table1 = table1.rename(columns={
    "count": "N",
    "mean": "Mean",
    "std": "Std Dev",
    "min": "Min",
    "max": "Max"
})

table1.index = [
    "Turnout (%)",
    "Log Population",
    "Log Income",
    "HS Education (%)",
    "African American (%)",
    "Urban (%)"
]
 
table1 = table1.round(2)

print("\nTable 1: Descriptive Statistics")
print(table1)


# Plotting


# Distribution Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(df["turnout_pct"], kde=True, ax=axes[0,0], color="steelblue")
axes[0,0].set_title("Turnout Distribution (%)")

sns.histplot(df["population"], kde=True, ax=axes[0,1], color="steelblue")
axes[0,1].set_title("Population")

sns.histplot(df["medfaminc"], kde=True, ax=axes[1,0], color="steelblue")
axes[1,0].set_title("Median Family Income")

sns.histplot(df["per_HSeducation"], kde=True, ax=axes[1,1], color="steelblue")
axes[1,1].set_title("HS Education Share")

plt.tight_layout()
plt.savefig("outputs/distributions.png", dpi=300)
plt.show()
plt.close()


# Registration
df["registration_label"] = df["registration"].map({
    0: "None",
    1: "Full",
    5: "Partial"
})

plt.figure(figsize=(6,4))

reg_means = df.groupby("registration_label")["turnout_pct"].mean().reset_index()

sns.barplot(
    x="registration_label",
    y="turnout_pct",
    hue="registration_label",
    data=reg_means,
    order=["None", "Partial", "Full"],
    palette=["steelblue", "darkorange", "seagreen"],
    legend=False
)

plt.title("Average Turnout by Registration Status")
plt.ylabel("Mean Turnout (%)")
plt.xlabel("Registration Type")

plt.tight_layout()
plt.savefig("outputs/turnout_by_registration_bar.png", dpi=300)
plt.show()
plt.close()


# Presidential Year
df["presyear_label"] = df["presyear"].map({
    0: "Non-Presidential Year",
    1: "Presidential Year"
})

plt.figure(figsize=(6,4))

pres_means = df.groupby("presyear_label")["turnout_pct"].mean().reset_index()

sns.barplot(
    x="presyear_label",
    y="turnout_pct",
    hue="presyear_label",
    data=pres_means,
    order=["Non-Presidential Year", "Presidential Year"],
    palette=["steelblue", "darkorange"],
    legend=False
)

plt.title("Average Turnout: Presidential vs Non-Presidential Years")
plt.ylabel("Mean Turnout (%)")
plt.xlabel("")

plt.tight_layout()
plt.savefig("outputs/turnout_by_presyear_bar.png", dpi=300)
plt.show()
plt.close()


# Law Change
df["lawchange_label"] = df["lawchange"].map({
    0: "Otherwise",
    1: "First Year After Law Change"
})

plt.figure(figsize=(6,4))

law_means = df.groupby("lawchange_label")["turnout_pct"].mean().reset_index()

sns.barplot(
    x="lawchange_label",
    y="turnout_pct",
    hue="lawchange_label",
    data=law_means,
    order=["Otherwise", "First Year After Law Change"],
    palette=["steelblue", "darkorange"],
    legend=False
)

plt.title("Average Turnout Around Law Changes")
plt.ylabel("Mean Turnout (%)")
plt.xlabel("")

plt.tight_layout()
plt.savefig("outputs/turnout_by_lawchange_bar.png", dpi=300)
plt.show()
plt.close()


# Income and Turnout
plt.figure(figsize=(6,4))
sns.scatterplot(x="log_inc", y="turnout", data=df, alpha=0.4, color="steelblue")

plt.title("Income vs Turnout")
plt.tight_layout()

plt.savefig("outputs/income_vs_turnout.png", dpi=300)
plt.show()
plt.close()


# Education and Turnout
plt.figure(figsize=(6,4))
sns.scatterplot(x="per_HSeducation", y="turnout", data=df, alpha=0.4, color="darkorange")

plt.title("Education vs Turnout")
plt.tight_layout()

plt.savefig("outputs/education_vs_turnout.png", dpi=300)
plt.show()
plt.close()


# State-level turnout
plt.figure(figsize=(6,4))

state_means = df.groupby("state")["turnout_pct"].mean().reset_index()

sns.barplot(
    x="state",
    y="turnout_pct",
    hue="state",
    data=state_means,
    palette=["steelblue", "darkorange"],
    legend=False
)

plt.title("Average Turnout by State")
plt.ylabel("Mean Turnout (%)")

plt.tight_layout()
plt.savefig("outputs/state_turnout.png", dpi=300)
plt.show()
plt.close()


# T-test (Law Change Effect)

df_clean = df[df["registration"] != 5].copy()

before = df_clean[df_clean["lawchange"] == 0]["turnout_pct"]
after  = df_clean[df_clean["lawchange"] == 1]["turnout_pct"]

t_stat, p_value = stats.ttest_ind(after, before, equal_var=False)

print("\nT-test results (Law Change Effect)\n")
print(f"Mean before: {before.mean():.2f}%")
print(f"Mean after : {after.mean():.2f}%")
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value    : {p_value:.3f}")