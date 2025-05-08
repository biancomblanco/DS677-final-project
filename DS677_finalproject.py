import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal
import torch_directml

# --- Load Dataset ---
df = pd.read_csv('C:/Users/Bianco Blanco/Downloads/bank.csv', sep=";")

# --- Exploratory Data Analysis (EDA) ---
# Basic info and statistics
print(df.info())
print(df.describe())

# Missing values per column
print("Missing values by column:\n", df.isnull().sum())

'''
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

Distrubtion of Numeric Fetures breakdown:

Distribution of Age is left tail skewed with a non-modal curve. This is interesting to see which says that typical
age of home buyers are younger

Distribution of Balance is heavily left tail skewed with outliers, however when we remove outliers the data still has a slight left
tail skew, however the data fits more of anormal curve with a clear peak around 100~300.

Distribution of day is rather sporadic and does not resemble a normal curve at all.

Distribution of duration is heavily left tail skewed with a non-modal curve. However, this might be best explained by the fact
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

# --- Device Setup ---
# First, attempt to use Apple Silicon (MPS) if available.
try:
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    else:
        raise ImportError
# If not Apple Silicon, catch and try AMD GPU on Windows via DirectML
except ImportError:
    try:
        dml = torch_directml.device()
        device = torch.device(dml)
        print("Using DirectML device:", device)
    except Exception:
        # Fallbacks: CUDA or CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA device")
        else:
            device = torch.device('cpu')
            print("Using CPU")

df = pd.read_csv('C:/Users/Bianco Blanco/Downloads/bank-full.csv', sep=";")

# --- Prepare Data for Bayesian Neural Network (BNN) ---

target_col = 'y'

# Separate target and features
y_series = df[target_col]
features_df = df.drop(columns=[target_col])

# Handle missing values in features
features_df = features_df.fillna(features_df.mean(numeric_only=True))

# Encode categorical features
features_df = pd.get_dummies(features_df, drop_first=True)

# Convert all columns to numeric, coercing errors
features_df = features_df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

# Fill any remaining NaNs
features_df = features_df.fillna(features_df.mean())

# Diagnostic: print dtypes
print('Feature column dtypes after conversion:')
print(features_df.dtypes)

# Force numpy array to float32 to avoid object dtype
features_array = features_df.values.astype(np.float32)
print('Converted feature array dtype:', features_array.dtype)

# Create tensor
X = torch.from_numpy(features_array)

y_array = (y_series.astype('category').cat.codes.values 
           if y_series.dtype == object or pd.api.types.is_categorical_dtype(y_series) 
           else y_series.values)

# Convert target to float tensor
y = torch.tensor(y_array.astype(np.float32)).unsqueeze(1)

# DataLoader setup
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# --- Define Bayesian Linear Layer using pure torch ---
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_sigma = nn.Parameter(torch.full((out_features,), -3.0))
        self.prior = Normal(loc=0.0, scale=prior_std)

    def forward(self, x):
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        self.log_prior = self.prior.log_prob(weight).sum() + self.prior.log_prob(bias).sum()
        q_weight = Normal(self.weight_mu, weight_sigma)
        q_bias = Normal(self.bias_mu, bias_sigma)
        self.log_variational_posterior = q_weight.log_prob(weight).sum() + q_bias.log_prob(bias).sum()
        return F.linear(x, weight, bias)

class BayesianNN(nn.Module):
    def __init__(self, in_features, hidden_size=50):
        super().__init__()
        self.blinear1 = BayesianLinear(in_features, hidden_size)
        self.blinear2 = BayesianLinear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.blinear1(x))
        return self.blinear2(x)

# --- Training Loop (Bayes by Backprop) ---
model = BayesianNN(in_features=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1000

for epoch in range(1, num_epochs + 1):
    epoch_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        likelihood = Normal(preds, 1.0)
        log_likelihood = likelihood.log_prob(yb).sum()
        log_prior = model.blinear1.log_prior + model.blinear2.log_prior
        log_posterior = model.blinear1.log_variational_posterior + model.blinear2.log_variational_posterior
        loss = (log_posterior - log_prior) - log_likelihood
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}\tLoss: {epoch_loss:.2f}")