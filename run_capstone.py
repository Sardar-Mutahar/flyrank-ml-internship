import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv('D:/ML internship/flyrank-ml-internship/data/raw/content_refresh_anonymized.csv')
print('Data loaded:', df.shape)

# Create target
df['is_declining_label'] = df['trend_direction'] == 'down'

# Time-aware split based on content_age_days 80th percentile
age_threshold = df['content_age_days'].quantile(0.8)
train = df[df['content_age_days'] <= age_threshold].copy()
test = df[df['content_age_days'] > age_threshold].copy()
print('Train rows:', len(train), 'Test rows:', len(test))

# === Baseline ===
# freshness_tier_score: 0 (0-30), 1 (31-90), 2 (91-180), 3 (181+)
def freshness_tier_score(tier):
    if tier == '0-30': return 0
    elif tier == '31-90': return 1
    elif tier == '91-180': return 2
    elif tier == '181+': return 3
    return 0

# position_tier_score: 0 (top_3), 1 (striking), 2 (page_1), 3 (page_3_5), 4 (deep)
def position_tier_score(tier):
    if tier == 'top_3': return 0
    elif tier == 'striking': return 1
    elif tier == 'page_1': return 2
    elif tier == 'page_3_5': return 3
    elif tier == 'deep': return 4
    return 0

train['freshness_tier_score'] = train['freshness_tier'].apply(freshness_tier_score)
test['freshness_tier_score'] = test['freshness_tier'].apply(freshness_tier_score)
train['position_tier_score'] = train['position_tier'].apply(position_tier_score)
test['position_tier_score'] = test['position_tier'].apply(position_tier_score)

train['baseline_score'] = train['freshness_tier_score'] + train['position_tier_score']
test['baseline_score'] = test['freshness_tier_score'] + test['position_tier_score']

# Calculate baseline metrics on test split
y_test = test['is_declining_label'].values

# For baseline, use baseline_score as the ranking score (higher = more likely declining)
baseline_ap = average_precision_score(y_test, test['baseline_score'].values)
# Precision@50: rank by baseline_score, count how many top 50 are positive, divide by 50
test_sorted = test.copy().sort_values('baseline_score', ascending=False)
top50 = test_sorted.head(50)
if len(top50) >= 50:
    tp50 = top50['is_declining_label'].sum()
    precision_at_50 = tp50 / 50
else:
    precision_at_50 = tp50 / len(top50)

print('=== BASELINE ===')
print('Average Precision:', baseline_ap)
print('Precision@50:', precision_at_50)
print('Baseline score range test:', test['baseline_score'].min(), '-', test['baseline_score'].max())

# === Model ===
# Five leakage-safe features
features = ['impressions_90d', 'clicks_90d', 'ctr', 'avg_position', 'content_age_days']

# Train logistic regression
X_train = train[features].values
y_train = train['is_declining_label'].values
X_test = test[features].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(random_state=42, class_weight='balanced', n_jobs=-1)
lr.fit(X_train_scaled, y_train)

# Get decision function scores for ranking
y_score = lr.decision_function(X_test_scaled)

# Model metrics
model_ap = average_precision_score(y_test, y_score)
model_scores = pd.DataFrame({'score': y_score, 'label': y_test})
model_sorted = model_scores.sort_values('score', ascending=False)
top50_model = model_sorted.head(50)
if len(top50_model) >= 50:
    tp50_model = top50_model['label'].sum()
    precision_at_50_model = tp50_model / 50
else:
    precision_at_50_model = top50_model['label'].sum() / len(top50_model)

print('=== LOGISTIC REGRESSION MODEL ===')
print('Average Precision:', model_ap)
print('Precision@50:', precision_at_50_model)

# Coefficients
coefs = pd.DataFrame({
    'feature': features,
    'coefficient': lr.coef_[0]
})
coefs['abs_coef'] = coefs['coefficient'].abs()
coefs = coefs.sort_values('abs_coef', ascending=False)
print('Coefficients:')
for _, row in coefs.iterrows():
    print(f"  {row['feature']}: {row['coefficient']:.4f} (abs: {row['abs_coef']:.4f})")

# === Precision@K analysis ===
precisions_k = {}
for k in [20, 50, 100]:
    topk = model_sorted.head(k)
    if len(topk) > 0:
        precisions_k[k] = topk['label'].sum() / k
    else:
        precisions_k[k] = 0
print('Precision@K:', precisions_k)

# === Save results ===
with open('results_summary.txt', 'w') as f:
    f.write('Baseline AP: %.4f\n' % baseline_ap)
    f.write('Baseline Precision@50: %.4f\n' % precision_at_50)
    f.write('Model AP: %.4f\n' % model_ap)
    f.write('Model Precision@50: %.4f\n' % precision_at_50_model)
    f.write('Model beats baseline AP: %s\n' % ('YES' if model_ap > baseline_ap else 'NO'))
    f.write('Model beats baseline Precision@50: %s\n' % ('YES' if precision_at_50_model > precision_at_50 else 'NO'))
    f.write('Age threshold: %.1f\n' % age_threshold)
    f.write('Train rows: %d\n' % len(train))
    f.write('Test rows: %d\n' % len(test))

# Save coefficients
coefs.to_csv('coefs.csv', index=False)

# Save precision curve data
with open('precisions_k.txt', 'w') as f:
    for k in sorted(precisions_k.keys()):
        f.write('Precision@%d: %.4f\n' % (k, precisions_k[k]))

# Generate charts
# Chart 1: Model vs baseline ranking distribution
plt.figure(figsize=(10, 6))
baseline_sorted = test.copy().sort_values('baseline_score', ascending=False)
baseline_sorted['rank'] = range(len(baseline_sorted))
plt.plot(baseline_sorted['rank'], baseline_sorted['baseline_score'], label='Baseline score', alpha=0.7, color='blue')
plt.plot(baseline_sorted['rank'], baseline_sorted['is_declining_label'].values, label='Actual label', alpha=0.7, color='red')
plt.xlabel('Rank (sorted by score)')
plt.ylabel('Score / Label')
plt.title('Baseline Ranking on Test Split')
plt.legend()
plt.tight_layout()
plt.savefig('chart1_model_vs_baseline.png')
plt.close()

# Chart 2: Precision@K
ks = sorted(precisions_k.keys())
pks = [precisions_k[k] for k in ks]
plt.figure(figsize=(8, 5))
plt.plot(ks, pks, 'bo-')
for i, k in enumerate(ks):
    plt.annotate('P@%d=%.3f' % (k, pks[i]), (ks[i], pks[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('K')
plt.ylabel('Precision@K')
plt.title('Precision@K Curve')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chart2_precision_k_curve.png')
plt.close()

# Chart 3: Feature coefficients
plt.figure(figsize=(10, 6))
top_coefs = coefs.sort_values('abs_coef', ascending=True)
plt.barh(range(len(top_coefs)), top_coefs['abs_coef'], align='center')
plt.yticks(range(len(top_coefs)), top_coefs['feature'])
plt.xlabel('|Coefficient|')
plt.title('Feature Coefficient Magnitudes')
plt.tight_layout()
plt.savefig('chart3_feature_coefficients.png')
plt.close()

print('All charts saved.')
print()
print('=== FINAL COMPARISON ===')
print('| Method | Average Precision | Precision@50 |')
print('|---|---:|---:|')
print('| Week 4 baseline | %.4f | %.4f |' % (baseline_ap, precision_at_50))
print('| Logistic Regression | %.4f | %.4f |' % (model_ap, precision_at_50_model))
print('| Model beats baseline AP: %s' % ('YES' if model_ap > baseline_ap else 'NO'))
print('| Model beats baseline Precision@50: %s' % ('YES' if precision_at_50_model > precision_at_50 else 'NO'))