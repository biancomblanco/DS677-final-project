import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal, Bernoulli
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve 
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
import torch_directml

# --- Load Dataset ---
df = pd.read_csv('C:/Users/Bianco Blanco/Downloads/bank.csv', sep=";")

# --- Exploratory Data Analysis (EDA) ---
# Basic info and statistics
print("Dataframe Info")
print(df.info())

print("-"*120)

print("Summary Statistics")
print(df.describe())

print("-"*120)

# Missing values per column
print("Missing values by column:\n", df.isnull().sum()) 

print("~"*120)

print("Distribution plots, pairs plots, and heatmap")

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


print("~"*120)
print("Interquantile EDA")
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
print("="*120)
print("Create Mosaic chart")
# Mosaic chart for these variables
cols = ['age','balance','duration','campaign','pdays','previous']
fig, axes = plt.subplots(2, 3, figsize=(10, 6))

for ax, col in zip(axes.flatten(), cols):
    sns.histplot(df[col].dropna(), kde=True,
                 ax=ax, bins=40, element='step')
    ax.set_title(col)
    ax.set_xlabel('')
    if col in ['balance', 'duration', 'pdays', 'previous']:
        ax.set_yscale('log')         
        ax.set_ylabel('log‐count')
    else:
        ax.set_ylabel('count')

plt.tight_layout()
plt.savefig('dist_mosaic.pdf')      

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

print("="*120)
print("Set GPU device")
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

print("="*120)
print("Load in full dataset")
df = pd.read_csv('C:/Users/Bianco Blanco/Downloads/bank-full.csv', sep=";")

# keep track of the original row indices
all_indices = df.index.to_numpy()

# --- Prepare Data for Bayesian Neural Network (BNN) ---

target_col = 'y'

# Separate target and features
y_series = df[target_col].map({'no': 0, 'yes': 1}).astype(np.float32)
features_df = df.drop(columns=[target_col])

# Handle missing values in features
features_df = features_df.fillna(features_df.mean(numeric_only=True))

# Encode categorical features
features_df = pd.get_dummies(features_df, drop_first=True)

# Convert all columns to numeric, coercing errors
features_df = features_df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

# Fill any remaining NaNs
features_df = features_df.fillna(features_df.mean())

print("+"*120)

# Diagnostic: print dtypes
print('Feature column dtypes after conversion:')
print(features_df.dtypes)

# Force numpy array to float32 for error handaling
features_array = features_df.values.astype(np.float32)
print('Converted feature array dtype:', features_array.dtype)

print("+"*120)

# --- split train/validation  ---
X_train_np, X_val_np, y_train_np, y_val_np, idx_train, idx_val = train_test_split(
    features_array, y_series.values, all_indices,
    test_size=0.1,
    stratify=y_series,
    random_state=42
)

# --- Standardize features  ---
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np).astype(np.float32)
X_val_np   = scaler.transform(X_val_np).astype(np.float32)

# --- Print number of features (input dimensions) ---
print(f"Number of input features: {X_train_np.shape[1]}")

# Convert to tensors
X_train = torch.from_numpy(X_train_np)
y_train = torch.tensor(y_train_np).unsqueeze(1)
X_val   = torch.from_numpy(X_val_np)
y_val   = torch.tensor(y_val_np).unsqueeze(1)

# DataLoader setup
train_dataset = TensorDataset(X_train, y_train)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
num_batches   = len(train_loader)               # for KL scaling

val_dataset   = TensorDataset(X_val, y_val)    
val_loader    = DataLoader(val_dataset, batch_size=64)  

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
model = BayesianNN(in_features=X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000

print("*"*120)
print("Begin BNN Training")
for epoch in range(1, num_epochs + 1):
    epoch_loss = 0.0
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)

        log_prior  = model.blinear1.log_prior   + model.blinear2.log_prior
        log_varpos = model.blinear1.log_variational_posterior \
                   + model.blinear2.log_variational_posterior
        kl = (log_varpos - log_prior) / num_batches

        likelihood = Bernoulli(logits=preds)
        log_likelihood = likelihood.log_prob(yb).sum()

        loss = kl - log_likelihood
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 100 == 0:
        avg_batch_loss = epoch_loss / num_batches

        # training accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in train_loader:
                preds = (torch.sigmoid(model(xb)) > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.numel()
        train_acc = correct / total

        # validation accuracy
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = (torch.sigmoid(model(xb)) > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.numel()
        val_acc = correct / total

        print(f"Epoch {epoch:4d}  Avg Loss: {avg_batch_loss:.2f}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")
        print('-'*40)

# --- plot ROC curve on validation set ---
model.eval()
val_probs, val_targets = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        p = torch.sigmoid(model(xb)).cpu().numpy().flatten()
        val_probs.extend(p.tolist())
        val_targets.extend(yb.cpu().numpy().flatten().tolist())

fpr, tpr, _ = roc_curve(val_targets, val_probs)
roc_auc = auc(fpr, tpr)

bnn_train_acc = train_acc
bnn_val_acc = val_acc
bnn_auc = roc_auc

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], 'k--', label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Validation ROC Curve")
plt.legend(loc="lower right")
plt.show()

print("*"*120)
print("Monte Carlo uncertainty prediction")

# --- Monte Carlo predictive uncertainty (after training) ---
model.eval()
T = 100  # number of MC samples
mc_samples = []

with torch.no_grad():
    for _ in range(T):
        batch_probs = []
        for xb, _ in val_loader:
            probs = torch.sigmoid(model(xb))
            batch_probs.append(probs.cpu().numpy())
        mc_samples.append(np.concatenate(batch_probs, axis=0))
mc_mean = np.mean(mc_samples, axis=0)
mc_std  = np.std(mc_samples,  axis=0)

print("&"*120)
print("MC predictive uncertainty (std) mean:", mc_std.mean())
print("MC predictive uncertainty (std) max:",  mc_std.max())

# Convert target list to a flat array
val_targets = np.array(val_targets).flatten()  

# Error handle Flatten mc_std 
mc_std = mc_std.flatten()

# Compute the 10%‐rejection threshold
threshold = np.percentile(mc_std, 90)

# Build a 1D boolean mask
keep_mask = mc_std <= threshold               

# Apply mask to both probs and targets
selected_probs = mc_mean[keep_mask]
selected_true  = val_targets[keep_mask]
selected_pred  = (selected_probs > 0.5).astype(int)

# Compute metrics
sel_acc = accuracy_score(selected_true, selected_pred)
sel_auc = roc_auc_score(selected_true, selected_probs)

print("&"*120)
print(f"Selective Prediction -> Acc: {sel_acc:.3f}, AUC: {sel_auc:.3f}")

# --- align the raw DataFrame to val set using idx_val ---
val_df = df.loc[idx_val].copy()     
val_df['true'] = y_val_np            
val_df['pred'] = (mc_mean > 0.5).astype(int)  

# compute FPs and FNs
fp = val_df[(val_df['pred'] == 1) & (val_df['true'] == 0)]
fn = val_df[(val_df['pred'] == 0) & (val_df['true'] == 1)]

fp_rate = len(fp) / len(val_df)
fn_rate = len(fn) / len(val_df)

print("@"*120)
print(f"\nFalse Positives: {len(fp)} ({fp_rate:.2%})")
print("Example False Positives:")
print(fp.drop_duplicates().head())

print(f"\nFalse Negatives: {len(fn)} ({fn_rate:.2%})")
print("Example False Negatives:")
print(fn.drop_duplicates().head()) 


# Attach the raw predicted probabilities in order to calculate ROC/calibration
val_df['pred_prob'] = mc_mean

# Martial status bias analysis
female_mask = val_df['marital'] == 'married'
female_probs = val_df.loc[female_mask, 'pred_prob']
female_labels = val_df.loc[female_mask, 'true']
female_auc = roc_auc_score(female_labels, female_probs)

print("@"*120)
print(f"\nAUC on married subset: {female_auc:.4f} (Drop: {bnn_auc - female_auc:.4f})")

# --- Calibration curve (reliability diagram) ---

# Use the true labels and the raw probabilities
prob_true, prob_pred = calibration_curve(
    val_df['true'],
    val_df['pred_prob'],
    n_bins=10,
    strategy='uniform'
)

plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label='BNN')
plt.plot([0,1],[0,1],'k--', label='Perfectly calibrated')
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Reliability Diagram (BNN)")
plt.legend()
plt.grid(True)
plt.show()

# -- Define LogisticRegression to compare to BNN-- #
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# -- Define NeuralNetwork to compare to BNN -- #    
class StandardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=50):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))    

# -- Define shared training loop for LogisticRegression and NeuralNetwork -- #    
def train_model(model, train_loader, val_loader, device, epochs=1000, lr=0.001, model_name="Model"):
    print(f"\nTraining {model_name} \n" + "="*50)

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            # --- Evaluate on Training Set ---
            model.eval()
            train_true, train_pred = [], []
            with torch.no_grad():
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    preds = model(xb).cpu()
                    train_pred.extend((preds > 0.5).float().numpy().flatten())
                    train_true.extend(yb.numpy().flatten())
            train_acc = accuracy_score(train_true, train_pred)

            # --- Evaluate on Validation Set ---
            val_true, val_pred, val_scores = [], [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    preds = model(xb).cpu()
                    val_scores.extend(preds.numpy().flatten())
                    val_pred.extend((preds > 0.5).float().numpy().flatten())
                    val_true.extend(yb.numpy().flatten())
            val_acc = accuracy_score(val_true, val_pred)
            val_auc = roc_auc_score(val_true, val_scores)

            print(f"Epoch {epoch:4d}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}  AUC: {val_auc:.4f}")
            print('-' * 50)

    return train_acc, val_acc, val_auc

# Instantiate and train both models
logistic_model = LogisticRegression(X_train.shape[1])
nn_model = StandardNN(X_train.shape[1], hidden_dim=50)

logistic_train_acc, logistic_val_acc, logistic_auc = train_model(
    logistic_model, train_loader, val_loader, device, model_name="Logistic Regression")

nn_train_acc, nn_val_acc, nn_auc = train_model(
    nn_model, train_loader, val_loader, device, model_name="Standard Neural Network")


results = pd.DataFrame({
    "Model": ["Logistic Regression", "Standard Neural Network", "Bayesian Neural Network"],
    "Train Accuracy": [logistic_train_acc, nn_train_acc, bnn_train_acc],
    "Validation Accuracy": [logistic_val_acc, nn_val_acc, bnn_val_acc],
    "Validation AUC": [logistic_auc, nn_auc, bnn_auc]
})

print("\nFinal Comparison of Models:\n")
print(results.to_string(index=False))