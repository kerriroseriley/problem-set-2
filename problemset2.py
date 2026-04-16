
"""   
Kerri Riley
Program Evaluation
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ----------------------------
# Create output folder
# ----------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("turnout.csv")

print(df.shape)
print(df.info())
df.head()

# ----------------------------
# Missing data plot
# ----------------------------
missing = df.isnull().mean().sort_values(ascending=False)

plt.figure(figsize=(8,4))
missing[missing > 0].plot(kind='bar')
plt.title("Share of Missing Values by Variable")
plt.ylabel("Proportion Missing")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/missing_values.png", dpi=300)
plt.show()
plt.close()


# ----------------------------
# Descriptive stats
# ----------------------------
key_vars = [
    "turnout", "population", "medfaminc", "per_HSeducation",
    "per_AfricanAmerican", "per_urban", "median_age", "log_inc"
]

desc_stats = df[key_vars].describe().T
print(desc_stats)


# ----------------------------
# Distributions
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(df["turnout"], kde=True, ax=axes[0,0])
axes[0,0].set_title("Turnout Distribution")

sns.histplot(df["population"], kde=True, ax=axes[0,1])
axes[0,1].set_title("Population (Skewed)")

sns.histplot(df["medfaminc"], kde=True, ax=axes[1,0])
axes[1,0].set_title("Median Family Income")

sns.histplot(df["per_HSeducation"], kde=True, ax=axes[1,1])
axes[1,1].set_title("HS Education Share")

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/distributions.png", dpi=300)
plt.show()
plt.close()


# ----------------------------
# Registration boxplot
# ----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="registration", y="turnout", data=df)
plt.title("Turnout by Registration Regime")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/turnout_by_registration.png", dpi=300)
plt.show()
plt.close()


# ----------------------------
# Presidential election years
# ----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="presyear", y="turnout", data=df)
plt.title("Turnout: Presidential vs Non-Presidential Years")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/turnout_by_presyear.png", dpi=300)
plt.show()
plt.close()


# ----------------------------
# Law change effect
# ----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="lawchange", y="turnout", data=df)
plt.title("Turnout Around Law Changes")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/turnout_by_lawchange.png", dpi=300)
plt.show()
plt.close()


# ----------------------------
# Correlation heatmap
# ----------------------------
plt.figure(figsize=(10,6))
corr = df[key_vars].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Key Variables)")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png", dpi=300)
plt.show()
plt.close()


# ----------------------------
# Income vs education
# ----------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x="log_inc", y="per_HSeducation", data=df, alpha=0.4)
plt.title("Income vs Education")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/income_vs_education.png", dpi=300)
plt.show()
plt.close()


# ----------------------------
# State-level turnout
# ----------------------------
plt.figure(figsize=(6,4))
df.groupby("state")["turnout"].mean().plot(kind="bar")
plt.title("Average Turnout by State")
plt.ylabel("Mean Turnout")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/state_turnout.png", dpi=300)
plt.show()
plt.close()