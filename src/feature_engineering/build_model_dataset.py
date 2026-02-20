import pandas as pd # for data manipulation - your data manipulation engine.
from pathlib import Path # for handling file paths in a platform-independent way.
from sklearn.model_selection import train_test_split # for splitting the dataset into training and testing sets. prevents cheating (data leakage).
from sklearn.preprocessing import LabelEncoder # encoding categorical variables into numeric format, which is necessary for most machine learning algorithms.

# Models cannot understand strings like "CAD" or "Retail".

from data_generation.generate_dataset import PROJECT_ROOT

# ------------------
# Load Data
# ------------------

project_root = Path(__file__).resolve().parent[2] # gives the full path of this script. moves two folders up. This line calculates the project root directory by taking the current file's path, resolving it to an absolute path, and then navigating up two levels in the directory structure. This is a common way to ensure that you can access files relative to the project root regardless of where the script is run from.

data_path = project_root / "data" / "raw" / "enterprise_fraud_transactions.csv" 

df = pd.read_csv(data_path)

"""
We are loading raw transactional data.
In real banks, this would come from:
    Snowflake
    BigQuery
    Data lake
    Event streams
Here we simulate that.
"""
print("Data loaded successfully. Shape:", df.shape)

# ------------------
# Basic Cleaning
# ------------------

# Drop identifiers that should not be used for modeling
df = df.drop(columns=["transaction_id", "session_id"])
"""
Why?
Identifiers:
    Have no predictive meaning.
    Can cause data leakage.
    Might create accidental correlations.
Models should not learn from arbitrary IDs.
This is clean modeling hygiene.
"""

# encode categorical variables
categorical_cols = ["txn_currency", "merchant_category"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

"""
If: txn_currency = ["CAD", "USD", "EUR"]
Then: txn_currency_encoded = [0, 1, 2]
This allows models to process categorical data as numbers.
"""

"""
⚠ Important Professional Insight
LabelEncoding assumes an artificial order.
    For Logistic Regression this is not ideal.
    For tree-based models (Random Forest, XGBoost), it's fine.

In production:
    We would use OneHotEncoding.
    Or target encoding.
    Or embeddings.
But for now, this is acceptable and practical.
"""



# ------------------
# Train Test Split
# ------------------

x = df.drop("fraud_label", axis=1)
y = df["fraud_label"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

print("Train.shape:", X_train.shape)
print("Test.shape:", X_test.shape)

# -----------------------------
# SAVE PROCESSED DATA
# -----------------------------

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

X_train.to_csv(PROCESSED_PATH / "X_train.csv", index=False)
X_test.to_csv(PROCESSED_PATH / "X_test.csv", index=False)
y_train.to_csv(PROCESSED_PATH / "y_train.csv", index=False)
y_test.to_csv(PROCESSED_PATH / "y_test.csv", index=False)

print("Processed datasets saved.")



