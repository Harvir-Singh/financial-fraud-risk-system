import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from pathlib import Path
import os

np.random.seed(42)
random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
N = 300000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "enterprise_fraud_transactions.csv"
print("Saving to:", OUTPUT_PATH)

# -----------------------------
# BASIC IDS
# -----------------------------
transaction_id = np.arange(1, N + 1)
user_id = np.random.randint(1000, 40000, N)
device_id = np.random.randint(2000, 15000, N)
merchant_id = np.random.randint(3000, 3600, N)
session_id = np.random.randint(5000, 500000, N)

# -----------------------------
# TIME FEATURES
# -----------------------------
start_date = datetime(2024, 1, 1)
txn_time = [start_date + timedelta(minutes=random.randint(0, 60*24*180)) for _ in range(N)]
txn_time = pd.to_datetime(txn_time)

txn_hour = [t.hour for t in txn_time]
txn_day_of_week = [t.weekday() for t in txn_time]
is_weekend_flag = [1 if d >= 5 else 0 for d in txn_day_of_week]

# -----------------------------
# TRANSACTION FEATURES
# -----------------------------
txn_amount = np.random.exponential(scale=85, size=N)
txn_currency = np.random.choice(["CAD", "USD", "EUR"], N, p=[0.7,0.2,0.1])
merchant_category = np.random.choice(["Retail","Electronics","Travel","Gaming","Crypto"], N)

high_amount_flag = (txn_amount > 300).astype(int)
cross_border_flag = (txn_currency != "CAD").astype(int)

# -----------------------------
# ACCOUNT FEATURES
# -----------------------------
account_age_days = np.random.exponential(scale=400, size=N)
kyc_level = np.random.choice([1,2,3], N, p=[0.2,0.5,0.3])
credit_limit = np.random.normal(5000, 2000, N)
credit_utilization_ratio = np.clip(np.random.normal(0.4,0.2,N),0,1)
past_fraud_flag = np.random.binomial(1, 0.05, N)
avg_txn_amount_90d = np.random.exponential(scale=70, size=N)
dormant_account_flag = np.random.binomial(1, 0.08, N)
new_payee_added_24h = np.random.binomial(1, 0.04, N)
email_change_7d = np.random.binomial(1, 0.03, N)
phone_change_7d = np.random.binomial(1, 0.02, N)

# -----------------------------
# VELOCITY FEATURES
# -----------------------------
txn_count_1h = np.random.poisson(0.5, N)
txn_count_24h = np.random.poisson(3, N)
txn_count_7d = np.random.poisson(15, N)
sum_amount_24h = txn_count_24h * np.random.exponential(60, N)
distinct_merchant_24h = np.random.poisson(2, N)
failed_login_count_1h = np.random.poisson(0.3, N)
password_reset_24h = np.random.binomial(1, 0.05, N)
address_change_7d = np.random.binomial(1, 0.03, N)
device_change_7d = np.random.binomial(1, 0.06, N)
rapid_fire_flag = (txn_count_1h > 3).astype(int)

# -----------------------------
# DEVICE FEATURES
# -----------------------------
device_user_count = np.random.randint(1, 8, N)
device_first_seen_days = np.random.exponential(200, N)
device_risk_score = np.random.uniform(0,1,N)
emulator_flag = np.random.binomial(1, 0.03, N)
jailbroken_flag = np.random.binomial(1, 0.02, N)
proxy_flag = np.random.binomial(1, 0.05, N)
ip_risk_score = np.random.uniform(0,1,N)
browser_change_flag = np.random.binomial(1, 0.04, N)
fingerprint_mismatch_flag = np.random.binomial(1, 0.03, N)
shared_device_cluster_size = np.random.randint(1, 10, N)

# -----------------------------
# GEO / BEHAVIORAL
# -----------------------------
geo_distance_last_login = np.random.exponential(50, N)
impossible_travel_flag = (geo_distance_last_login > 500).astype(int)
login_hour_deviation = np.random.normal(0,2,N)
session_duration_zscore = np.random.normal(0,1,N)

# -----------------------------
# AMOUNT ZSCORE
# -----------------------------
txn_amount_zscore = (txn_amount - avg_txn_amount_90d) / (avg_txn_amount_90d + 1)

# -----------------------------
# FRAUD PROBABILITY LOGIC
# -----------------------------
fraud_probability = (
    0.008
    + 0.07 * high_amount_flag
    + 0.09 * (account_age_days < 30)
    + 0.08 * rapid_fire_flag
    + 0.06 * (device_user_count > 4)
    + 0.05 * impossible_travel_flag
    + 0.05 * (failed_login_count_1h > 2)
    + 0.04 * past_fraud_flag
    + 0.04 * proxy_flag
    + 0.03 * cross_border_flag
)

fraud_probability = np.clip(fraud_probability, 0, 0.5)
fraud_label = np.random.binomial(1, fraud_probability)

# -----------------------------
# BUILD DATAFRAME
# -----------------------------
df = pd.DataFrame({
    "transaction_id": transaction_id,
    "user_id": user_id,
    "device_id": device_id,
    "merchant_id": merchant_id,
    "session_id": session_id,
    "txn_time": txn_time,
    "txn_amount": txn_amount,
    "txn_currency": txn_currency,
    "txn_hour": txn_hour,
    "txn_day_of_week": txn_day_of_week,
    "is_weekend_flag": is_weekend_flag,
    "high_amount_flag": high_amount_flag,
    "txn_amount_zscore": txn_amount_zscore,
    "cross_border_flag": cross_border_flag,
    "merchant_category": merchant_category,
    "txn_count_1h": txn_count_1h,
    "txn_count_24h": txn_count_24h,
    "txn_count_7d": txn_count_7d,
    "sum_amount_24h": sum_amount_24h,
    "distinct_merchant_24h": distinct_merchant_24h,
    "failed_login_count_1h": failed_login_count_1h,
    "password_reset_24h": password_reset_24h,
    "address_change_7d": address_change_7d,
    "device_change_7d": device_change_7d,
    "rapid_fire_flag": rapid_fire_flag,
    "device_user_count": device_user_count,
    "device_first_seen_days": device_first_seen_days,
    "device_risk_score": device_risk_score,
    "emulator_flag": emulator_flag,
    "jailbroken_flag": jailbroken_flag,
    "proxy_flag": proxy_flag,
    "ip_risk_score": ip_risk_score,
    "browser_change_flag": browser_change_flag,
    "fingerprint_mismatch_flag": fingerprint_mismatch_flag,
    "shared_device_cluster_size": shared_device_cluster_size,
    "account_age_days": account_age_days,
    "kyc_level": kyc_level,
    "credit_limit": credit_limit,
    "credit_utilization_ratio": credit_utilization_ratio,
    "past_fraud_flag": past_fraud_flag,
    "avg_txn_amount_90d": avg_txn_amount_90d,
    "dormant_account_flag": dormant_account_flag,
    "new_payee_added_24h": new_payee_added_24h,
    "email_change_7d": email_change_7d,
    "phone_change_7d": phone_change_7d,
    "geo_distance_last_login": geo_distance_last_login,
    "impossible_travel_flag": impossible_travel_flag,
    "login_hour_deviation": login_hour_deviation,
    "session_duration_zscore": session_duration_zscore,
    "fraud_label": fraud_label
})

# -----------------------------
# SAVE
# -----------------------------
os.makedirs(PROJECT_ROOT / "data" / "raw", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Enterprise dataset generated.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Fraud Rate:", round(df["fraud_label"].mean()*100,2), "%")
print("Saved file to:", os.path.abspath(OUTPUT_PATH))