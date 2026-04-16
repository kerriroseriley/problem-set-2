"""
Kerri Riley
Program Evaluation 
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")


# Load data

df = pd.read_csv("turnout.csv")


# Convert turnout to percent 
df["turnout_pct"] = df["turnout"] * 100

# Handle missing data 
missing = df.isnull().mean().sort_values(ascending=False)



def codebook(df):
    print("\n" + "="*80)
    print("CODEBOOK SUMMARY")
    print("="*80)

    for col in df.columns:
        print("\n")
        print(f"Variable: {col}")
        print("\n")

        # Basic info
        print(f"Type: {df[col].dtype}")
        print(f"Observations: {len(df)}")
        print(f"Missing: {df[col].isna().sum()} ({df[col].isna().mean():.2%})")
        print(f"Unique values: {df[col].nunique()}")

        # Numeric variables
        if pd.api.types.is_numeric_dtype(df[col]):
            print("\nSummary statistics:")
            desc = df[col].describe(percentiles=[.1, .25, .5, .75, .9])
            print(desc)

        # Categorical variables
        else:
            print("\nTop categories:")
            print(df[col].value_counts().head(10))


codebook(df)


plt.figure(figsize=(8,4))
missing[missing > 0].plot(kind='bar')
plt.title("Share of Missing Values by Variable")
plt.ylabel("Proportion Missing")
plt.tight_layout()

plt.savefig("outputs/missing_values.png", dpi=300)
plt.show()
plt.close()

# Key variables
key_vars = [
    "turnout_pct", "population", "medfaminc", "per_HSeducation",
    "per_AfricanAmerican", "per_urban", "median_age", "log_inc"
]


# Distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(df["turnout_pct"], kde=True, ax=axes[0,0])
axes[0,0].set_title("Turnout Distribution (%)")

sns.histplot(df["population"], kde=True, ax=axes[0,1])
axes[0,1].set_title("Population (Skewed)")

sns.histplot(df["medfaminc"], kde=True, ax=axes[1,0])
axes[1,0].set_title("Median Family Income")

sns.histplot(df["per_HSeducation"], kde=True, ax=axes[1,1])
axes[1,1].set_title("HS Education Share")

plt.tight_layout()

plt.savefig("outputs/distributions.png", dpi=300)
plt.show()
plt.close()

# Registration
plt.figure(figsize=(6,4))

reg_means = df.groupby("registration")["turnout_pct"].mean().reset_index()

sns.barplot(x="registration", y="turnout_pct", data=reg_means, palette="viridis")

plt.title("Average Turnout by Registration Regime")
plt.ylabel("Mean Turnout (%)")
plt.xlabel("Registration Code")

plt.tight_layout()
plt.savefig("outputs/turnout_by_registration_bar.png", dpi=300)
plt.show()
plt.close()

# Presidential Year
plt.figure(figsize=(6,4))

pres_means = df.groupby("presyear")["turnout_pct"].mean().reset_index()

sns.barplot(x="presyear", y="turnout_pct", data=pres_means, palette="Set2")

plt.title("Average Turnout: Presidential vs Non-Presidential Years")
plt.ylabel("Mean Turnout (%)")

plt.tight_layout()
plt.savefig("outputs/turnout_by_presyear_bar.png", dpi=300)
plt.show()
plt.close()

# Law Change
plt.figure(figsize=(6,4))

law_means = df.groupby("lawchange")["turnout_pct"].mean().reset_index()

sns.barplot(x="lawchange", y="turnout_pct", data=law_means, palette="Set1")

plt.title("Average Turnout Around Law Changes")
plt.ylabel("Mean Turnout (%)")

plt.tight_layout() 
plt.savefig("outputs/turnout_by_lawchange_bar.png", dpi=300)
plt.show()
plt.close()

# Correlation matric
plt.figure(figsize=(10,6))

corr = df[key_vars].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Matrix (Key Variables)")
plt.tight_layout()

plt.savefig("outputs/correlation_matrix.png", dpi=300)
plt.show()
plt.close()

# Income vs Education
plt.figure(figsize=(6,4))

sns.scatterplot(x="log_inc", y="per_HSeducation", data=df, alpha=0.4)

plt.title("Income vs Education")
plt.tight_layout()

plt.savefig("outputs/income_vs_education.png", dpi=300)
plt.show()
plt.close()

# State-level turnout
plt.figure(figsize=(6,4))

state_means = df.groupby("state")["turnout_pct"].mean().reset_index()

sns.barplot(x="state", y="turnout_pct", data=state_means)

plt.title("Average Turnout by State")
plt.ylabel("Mean Turnout (%)")

plt.tight_layout()
plt.savefig("outputs/state_turnout.png", dpi=300)
plt.show()
plt.close()

# Convert turnout to percent (optional but fine for interpretation)
df["turnout_pct"] = df["turnout"] * 100

# Exclude registration == 5
df_clean = df[df["registration"] != 5].copy()

before = df_clean[df_clean["lawchange"] == 0]["turnout_pct"]
after  = df_clean[df_clean["lawchange"] == 1]["turnout_pct"]

# Two-sample t-test (Welch's t-test by default unequal variances)
t_stat, p_value = stats.ttest_ind(after, before, equal_var=False)

print("\n£T-test results (Law Change Effect)")
print("-----------------------------------")
print(f"Mean before: {before.mean():.2f}%")
print(f"Mean after : {after.mean():.2f}%")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value    : {p_value:.6f}")