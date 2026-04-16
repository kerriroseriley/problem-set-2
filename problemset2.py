
"""
Kerri Riley
Program Evaluation


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Load data
df = pd.read_csv("your_dataset.csv")

# Quick overview
print(df.shape)
print(df.info())
df.head()

missing = df.isnull().mean().sort_values(ascending=False)

plt.figure(figsize=(8,4))
missing[missing > 0].plot(kind='bar')
plt.title("Share of Missing Values by Variable")
plt.ylabel("Proportion Missing")
plt.show()


# Descriptive Statistics
key_vars = [
    "turnout", "population", "medfaminc", "per_HSeducation",
    "per_AfricanAmerican", "per_urban", "median_age", "log_inc"
]

df[key_vars].describe().T


# Distribution of Key Outcomes and Predictors
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
plt.show()


# Turnout by Key CAtegorical Indicators
plt.figure(figsize=(6,4))
sns.boxplot(x="registration", y="turnout", data=df)
plt.title("Turnout by Registration Regime")
plt.show()


# Presidential Election Years
sns.boxplot(x="presyear", y="turnout", data=df)
plt.title("Turnout: Presidential vs Non-Presidential Years")
plt.show()



# Law Change Effect
sns.boxplot(x="lawchange", y="turnout", data=df)
plt.title("Turnout Around Law Changes")
plt.show()

# Correlation Structure
plt.figure(figsize=(10,6))
corr = df[key_vars].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Key Variables)")
plt.show()


# Income vs Education Relationship
plt.figure(figsize=(6,4))
sns.scatterplot(x="log_inc", y="per_HSeducation", data=df, alpha=0.4)
plt.title("Income vs Education")
plt.show()

# State Level Turnout Comparison
df.groupby("state")["turnout"].mean().plot(kind="bar", figsize=(5,3))
plt.title("Average Turnout by State")
plt.ylabel("Mean Turnout")
plt.show()