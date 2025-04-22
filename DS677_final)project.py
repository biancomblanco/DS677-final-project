import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Load Dataset ---
df = pd.read_csv('C:/Users/Bianco Blanco/Downloads/bank.csv', sep=";")
# --- Exploratory Data Analysis (EDA) ---
# Basic info and statistics
print(df.info())
print(df.describe())

# Missing values per column
print("Missing values by column:\n", df.isnull().sum())

# Distribution plots for numeric features
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Pairwise relationships
sns.pairplot(df)
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(12,10))
corr = df.select_dtypes(include=['int64', 'float64']).corr()

sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

Q1 = df['balance'].quantile(0.25)
Q3 = df['balance'].quantile(0.75)
IQR = Q3 - Q1

# Keep rows within 1.5 * IQR
df_filtered = df[(df['balance'] >= Q1 - 1.5 * IQR) & (df['balance'] <= Q3 + 1.5 * IQR)]

# Plot filtered distribution
plt.figure()
sns.histplot(df_filtered['balance'], kde=True)
plt.title('Filtered Distribution of Balance (Outliers Removed)')
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.show()



'''
Distrubtion of Numeric Fetures breakdown:

Distribution of Age is left tail skewed with a non-modal curve. This is interesting to see which says that typical
age of home buyers are younger

Distribution of Balance is heavily left tail skewed with outliers, however when we remove outliers the data still has a slight left
tail skew, however the data fits more of anormal curve with a clear peak around 100~300.

Distribution of day is rather sporadic and does not resemble a normal curve at all.

Distribuution of duration is heavily left tail skewed with a non-modal curve. However, this might be best explained by the fact
that as marketers they want to keep this distribution as close to 0.

Distribution of campaign is heafvily left tail skewed with a non-modal curve. However, this might be best explained by the fact
that as marketers they have an ideal amount of times someone should be contacted and also perceived mariginal returns on each subsequent
outreach attempt.

Distribution of pdays is nowhere close to being a normal curve. However, this is a marketing column used to track when is the last time
someone has reached out toe contact this opportunity.

Distribution of previous is nowhere close to being a normal curve. However, this is a marketing column used to track how many times a client
has been contacted in previous campaign.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Looking at the pairs chart there is not much interesting relationships out there. They are either seem stacked in one direction or
the plot's variable are inherently hard to draw much information from.

Looking at the feature correlation matrix it follows the same pattern.

'''