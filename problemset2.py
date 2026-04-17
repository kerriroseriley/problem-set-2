"""
Kerri Riley
Program Evaluation 
"""

# Import needed modules
import pandas as pd              # data manipulation
import matplotlib.pyplot as plt  # plotting
import seaborn as sns            # nicer visualizations
from scipy import stats          # statistical tests

# Set default plot style
sns.set(style="whitegrid", palette="Set2")


# Load the data
df = pd.read_csv("turnout.csv")

# Convert turnout from proportion to percent
df["turnout_pct"] = df["turnout"] * 100

# Check proportion of missing values in each column
missing = df.isnull().mean().sort_values(ascending=False)


# Codebook Function
def codebook(df):
    """Prints a summary of each variable in the dataset."""
    print("/nCodebook Summary/n")
    
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
        else:
            # Categorical variables
            print("\nTop categories:")
            print(df[col].value_counts().head(10))

# codebook(df)  # Uncomment to run


# Descriptive Statistics
vars_table1 = [
    "turnout_pct",
    "log_pop",
    "log_inc",
    "per_HSeducation",
    "per_AfricanAmerican",
    "per_urban"
]

# Generate summary stats
table1 = df[vars_table1].describe().T

# Keep only key columns
table1 = table1[["count", "mean", "std", "min", "max"]]

# Rename columns for readability
table1 = table1.rename(columns={
    "count": "N",
    "mean": "Mean",
    "std": "Std Dev",
    "min": "Min",
    "max": "Max"
})

# Rename row labels
table1.index = [
    "Turnout (%)",
    "Log Population",
    "Log Income",
    "HS Education (%)",
    "African American (%)",
    "Urban (%)"
]

# Round values
table1 = table1.round(2)

print("\nTable 1: Descriptive Statistics")
print(table1)


# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Turnout distribution
sns.histplot(df["turnout_pct"], kde=True, ax=axes[0,0], color="steelblue")
axes[0,0].set_title("Turnout Distribution (%)")

# Population distribution
sns.histplot(df["population"], kde=True, ax=axes[0,1], color="steelblue")
axes[0,1].set_title("Population")

# Income distribution
sns.histplot(df["medfaminc"], kde=True, ax=axes[1,0], color="steelblue")
axes[1,0].set_title("Median Family Income")

# Education distribution
sns.histplot(df["per_HSeducation"], kde=True, ax=axes[1,1], color="steelblue")
axes[1,1].set_title("HS Education Share")

plt.tight_layout()
plt.savefig("outputs/distributions.png", dpi=300)
plt.show()
plt.close()


# Registration vs Turnout
# Recode numeric variable into labels
df["registration_label"] = df["registration"].map({
    0: "None",
    1: "Full",
    5: "Partial"
})

plt.figure(figsize=(6,4))

# Compute mean turnout by category
reg_means = df.groupby("registration_label")["turnout_pct"].mean().reset_index()

# Bar plot (fixed Seaborn warning using hue)
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


# Presidential vs non-presidential years

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



# Law change effect 
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


# Scatter plots (relationships)

# Income vs turnout
plt.figure(figsize=(6,4))
sns.scatterplot(x="log_inc", y="turnout", data=df, alpha=0.4, color="steelblue")

plt.title("Income vs Turnout")
plt.tight_layout()

plt.savefig("outputs/income_vs_turnout.png", dpi=300)
plt.show()
plt.close()

# Education vs turnout
plt.figure(figsize=(6,4))
sns.scatterplot(x="per_HSeducation", y="turnout", data=df, alpha=0.4, color="darkorange")

plt.title("Education vs Turnout")
plt.tight_layout()

plt.savefig("outputs/education_vs_turnout.png", dpi=300)
plt.show()
plt.close()


# State-level turnout comparison
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



# T-test: effect of law change

# Drop "partial" registration group for cleaner comparison
df_clean = df[df["registration"] != 5].copy()

# Split into before vs after groups
before = df_clean[df_clean["lawchange"] == 0]["turnout_pct"]
after  = df_clean[df_clean["lawchange"] == 1]["turnout_pct"]

# Run two-sample t-test (unequal variance)
t_stat, p_value = stats.ttest_ind(after, before, equal_var=False)

# Print results
print("\nT-test results (Law Change Effect)\n")
print(f"Mean before: {before.mean():.2f}%")
print(f"Mean after : {after.mean():.2f}%")
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value    : {p_value:.3f}")