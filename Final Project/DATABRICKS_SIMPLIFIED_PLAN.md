# Databricks Implementation - Simplified Plan (User Summary Only)

**Course:** DAMO630 - Advanced Data Analytics
**Approach:** Upload pre-aggregated `sampled_users_summary.csv` to Databricks
**Status:** ✅ Local Complete | 🚀 Ready for Cloud Scale

---

## 🎯 Strategy: Use Pre-Computed User Summary

### What We're Uploading:
- ✅ **ONLY** `sampled_users_summary.csv` (252K users, all features pre-computed locally)
- ❌ **NOT** uploading Oct/Nov raw event CSVs (too large, already aggregated)

### What We'll Demonstrate in Databricks:
1. **Survival Analysis at Scale** - Kaplan-Meier, Cox model, risk scoring (using Spark + pandas)
2. **Large-Scale Mining** - RFM segmentation, cohort analysis (pure Spark SQL)
3. **Cloud Infrastructure** - Delta tables, distributed processing, dashboards

### ⚠️ **Limitation:**
- **Recommendation System** requires event-level data (user-item interactions)
- **Solution:** Either (a) skip recommendations in Databricks OR (b) use local recommendations + upload results

---

## 📁 Single File to Upload

### `sampled_users_summary.csv`
**Size:** ~25 MB
**Rows:** 252,899 users
**Columns:** 18 (all pre-computed from local pandas implementation)

```
Schema:
├── user_id (int)
├── total_events (int)
├── n_sessions (int)
├── n_views (int)
├── n_carts (int)
├── n_purchases (int)
├── total_spent (float)
├── unique_products_viewed (int)
├── unique_brands (int)
├── unique_categories (int)
├── first_event (datetime)
├── last_event (datetime)
├── tenure_days (int)
├── avg_events_per_session (float)
├── conversion_rate (float)
├── cart_to_purchase_rate (float)
├── avg_order_value (float)
└── user_strata (str)
```

---

## 🔧 Phase 1: Upload & Setup (10 minutes)

### Step 1.1: Create Databricks Workspace
```
URL: https://community.cloud.databricks.com/
Cluster: 4 cores, 14GB RAM (Standard_DS3_v2)
Runtime: 13.3 LTS (Spark 3.4.1)
```

### Step 1.2: Upload File via UI
1. Go to **Data** → **Add Data** → **Upload File**
2. Upload `sampled_users_summary.csv`
3. Target location: `dbfs:/FileStore/final_project/sampled_users_summary.csv`

### Step 1.3: Load into Spark (Notebook Cell 1)

```python
# Databricks Notebook: 01_Load_Data

from pyspark.sql import functions as F

# Load user summary
user_summary = spark.read.csv(
    "dbfs:/FileStore/final_project/sampled_users_summary.csv",
    header=True,
    inferSchema=True
)

# Fix datetime columns (if loaded as strings)
user_summary = user_summary \
    .withColumn("first_event", F.to_timestamp("first_event")) \
    .withColumn("last_event", F.to_timestamp("last_event"))

# Verify schema
user_summary.printSchema()

# Basic stats
print(f"✅ Total users loaded: {user_summary.count():,}")
user_summary.groupBy("user_strata").count().orderBy(F.desc("count")).show()

# Cache for performance
user_summary.cache()

# Save as Delta table
user_summary.write.format("delta").mode("overwrite").saveAsTable("default.user_summary")
print("✅ Saved as Delta table: default.user_summary")
```

**Output:**
```
✅ Total users loaded: 252,899
+----------------+------+
|    user_strata | count|
+----------------+------+
|   light_browser|xxx,xxx|
|  heavy_browser |xxx,xxx|
|         buyer  |xxx,xxx|
|cart_abandoner  |xxx,xxx|
|high_value_buyer|xxx,xxx|
|   power_buyer  |xxx,xxx|
+----------------+------+

✅ Saved as Delta table: default.user_summary
```

---

## 🔧 Phase 2: Survival Analysis (30 minutes)

### Step 2.1: Create Survival Dataset (Spark SQL)

```python
# Databricks Notebook: 02_Survival_Analysis

from pyspark.sql import functions as F

observation_end = F.lit("2019-11-30").cast("timestamp")
churn_threshold_days = 30

# Filter to buyers only
buyers = spark.table("default.user_summary").filter(F.col("n_purchases") > 0)

# Calculate survival metrics
survival_data = buyers \
    .withColumn("survival_time", F.col("tenure_days")) \
    .withColumn("days_since_last", F.datediff(observation_end, F.col("last_event"))) \
    .withColumn("event", (F.col("days_since_last") > churn_threshold_days).cast("int"))

# Save as Delta table
survival_data.write.format("delta").mode("overwrite").saveAsTable("default.survival_data")

# Statistics
total = survival_data.count()
churned = survival_data.filter(F.col("event") == 1).count()
active = survival_data.filter(F.col("event") == 0).count()

print(f"\n📊 Survival Dataset:")
print(f"   Total customers: {total:,}")
print(f"   Churned (event=1): {churned:,} ({churned/total*100:.1f}%)")
print(f"   Active (event=0): {active:,} ({active/total*100:.1f}%)")

display(survival_data.limit(10))
```

### Step 2.2: Kaplan-Meier & Cox Model (Convert to Pandas)

```python
# Install lifelines
%pip install lifelines

# Convert to pandas for survival modeling
survival_pdf = survival_data.toPandas()

from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt

# Kaplan-Meier Survival Curve
kmf = KaplanMeierFitter()
kmf.fit(
    durations=survival_pdf['survival_time'],
    event_observed=survival_pdf['event'],
    label='All Customers'
)

fig, ax = plt.subplots(figsize=(12, 6))
kmf.plot_survival_function(ax=ax, ci_show=True)
ax.set_title('Customer Survival Curve (Databricks Cloud)', fontsize=14, fontweight='bold')
ax.set_xlabel('Days Since First Purchase')
ax.set_ylabel('Survival Probability')
ax.grid(True, alpha=0.3)
display(fig)

print(f"Median survival time: {kmf.median_survival_time_:.1f} days")
```

### Step 2.3: Cox Model (Actual Column Names)

```python
from sklearn.preprocessing import StandardScaler

# VALIDATED features from local implementation
cox_features = [
    'n_purchases',
    'total_spent',
    'avg_order_value',
    'n_sessions',
    'cart_to_purchase_rate',
    'conversion_rate',
    'tenure_days',
    'unique_products_viewed',
    'avg_events_per_session'
]

# Prepare data
cox_data = survival_pdf[cox_features + ['survival_time', 'event']].copy()
cox_data = cox_data.replace([float('inf'), float('-inf')], float('nan'))
cox_data = cox_data.dropna()

# Normalize features
scaler = StandardScaler()
cox_data[cox_features] = scaler.fit_transform(cox_data[cox_features])

# Fit Cox model
cph = CoxPHFitter(penalizer=0.01)
cph.fit(cox_data, duration_col='survival_time', event_col='event')

# Display results
cph.print_summary()

# Plot hazard ratios
fig, ax = plt.subplots(figsize=(10, 8))
cph.plot(ax=ax)
ax.set_title('Hazard Ratios - Churn Risk Factors (Databricks)', fontsize=14, fontweight='bold')
ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
ax.legend()
display(fig)

# Model performance
from lifelines.utils import concordance_index
c_index = concordance_index(
    cox_data['survival_time'],
    -cph.predict_partial_hazard(cox_data[cox_features]),
    cox_data['event']
)
print(f"\n✅ C-index: {c_index:.3f} {'(Good)' if c_index > 0.7 else '(Moderate)'}")
```

### Step 2.4: Risk Scoring & Save Back to Spark

```python
# Prepare all features
all_features = survival_pdf[cox_features].copy()
all_features = all_features.replace([float('inf'), float('-inf')], float('nan'))
all_features = all_features.fillna(all_features.median())

# Scale features
all_features_scaled = scaler.transform(all_features)

# Predict risk scores
survival_pdf['risk_score'] = cph.predict_partial_hazard(all_features_scaled)

# Normalize to 0-100
risk_min, risk_max = survival_pdf['risk_score'].min(), survival_pdf['risk_score'].max()
survival_pdf['risk_score_normalized'] = (
    (survival_pdf['risk_score'] - risk_min) / (risk_max - risk_min) * 100
)

# Segment customers
import pandas as pd
survival_pdf['risk_segment'] = pd.cut(
    survival_pdf['risk_score_normalized'],
    bins=[0, 33, 67, 100],
    labels=['Low Risk', 'Medium Risk', 'High Risk'],
    include_lowest=True
)

# Time-to-churn predictions
time_points = [7, 14, 30, 60, 90]
surv_funcs = cph.predict_survival_function(all_features_scaled, times=time_points)

for t in time_points:
    survival_pdf[f'survival_prob_{t}d'] = surv_funcs.loc[t].values
    survival_pdf[f'churn_prob_{t}d'] = 1 - surv_funcs.loc[t].values

# Convert back to Spark
survival_scored = spark.createDataFrame(survival_pdf)

# Save to Delta table
survival_scored.write.format("delta").mode("overwrite").saveAsTable("default.customer_risk_scores")

print(f"\n✅ Risk scores saved to Delta table")

# Display risk summary
survival_scored.groupBy("risk_segment").agg(
    F.count("*").alias("customers"),
    F.sum("total_spent").alias("revenue_at_risk"),
    F.mean("n_purchases").alias("avg_purchases"),
    F.mean("event").alias("churn_rate")
).orderBy(F.desc("revenue_at_risk")).show()
```

### Step 2.5: Export High-Risk Customers

```python
# Identify urgent intervention cases
urgent = survival_pdf[
    (survival_pdf['risk_segment'] == 'High Risk') &
    (survival_pdf['churn_prob_30d'] > 0.5)
][['user_id', 'user_strata', 'risk_score_normalized', 'churn_prob_7d', 'churn_prob_30d', 'total_spent']]

print(f"\n🚨 Urgent Interventions:")
print(f"   Customers: {len(urgent):,}")
print(f"   Revenue at risk: ${urgent['total_spent'].sum():,.2f}")

# Export to DBFS
urgent.to_csv('/dbfs/FileStore/final_project/high_risk_interventions.csv', index=False)
print(f"✅ Exported to DBFS")

display(urgent.head(10))
```

---

## 🔧 Phase 3: Large-Scale Data Mining (Pure Spark)

### Step 3.1: RFM Segmentation (Distributed)

```python
# Databricks Notebook: 03_Data_Mining

from pyspark.sql import functions as F
from pyspark.sql.window import Window

observation_date = F.lit("2019-11-30").cast("timestamp")

# Calculate RFM
rfm = spark.table("default.user_summary") \
    .filter(F.col("n_purchases") > 0) \
    .withColumn("recency", F.datediff(observation_date, F.col("last_event"))) \
    .withColumn("frequency", F.col("n_purchases")) \
    .withColumn("monetary", F.col("total_spent"))

# Calculate quintile scores (1-5)
rfm = rfm.withColumn("R_score", F.expr("ntile(5) OVER (ORDER BY recency DESC)")) \
         .withColumn("F_score", F.expr("ntile(5) OVER (ORDER BY frequency ASC)")) \
         .withColumn("M_score", F.expr("ntile(5) OVER (ORDER BY monetary ASC)"))

# Combine RFM score
rfm = rfm.withColumn("RFM_Score", F.col("R_score") + F.col("F_score") + F.col("M_score"))

# Label segments
rfm = rfm.withColumn(
    "RFM_Label",
    F.when(F.col("RFM_Score") >= 13, "Champions")
     .when(F.col("RFM_Score") >= 10, "Loyal Customers")
     .when(F.col("RFM_Score") >= 7, "Potential Loyalists")
     .when(F.col("RFM_Score") >= 5, "At Risk")
     .otherwise("Lost")
)

# Save to Delta table
rfm.write.format("delta").mode("overwrite").saveAsTable("default.rfm_segments")

# Summary
print("\n📊 RFM Segment Summary:")
rfm.groupBy("RFM_Label").agg(
    F.count("*").alias("customers"),
    F.sum("monetary").alias("total_revenue"),
    F.mean("recency").alias("avg_recency"),
    F.mean("frequency").alias("avg_frequency")
).orderBy(F.desc("total_revenue")).show()
```

### Step 3.2: Cohort Analysis (User Strata Performance)

```python
# Cohort analysis by user strata
cohort_analysis = spark.table("default.user_summary").groupBy("user_strata").agg(
    F.count("*").alias("customers"),
    F.sum("total_spent").alias("total_revenue"),
    F.mean("total_spent").alias("avg_revenue_per_user"),
    F.mean("n_purchases").alias("avg_purchases"),
    F.mean("conversion_rate").alias("avg_conversion_rate"),
    F.mean("cart_to_purchase_rate").alias("avg_cart_conversion"),
    F.mean("tenure_days").alias("avg_lifetime_days")
).orderBy(F.desc("total_revenue"))

print("\n📊 Cohort Analysis by User Strata:")
display(cohort_analysis)

# Save to Delta table
cohort_analysis.write.format("delta").mode("overwrite").saveAsTable("default.cohort_analysis")
```

### Step 3.3: Customer Lifetime Value (CLV) Analysis

```python
# Calculate CLV metrics
clv_analysis = spark.table("default.user_summary") \
    .filter(F.col("n_purchases") > 0) \
    .withColumn("avg_purchase_value", F.col("total_spent") / F.col("n_purchases")) \
    .withColumn("purchase_frequency", F.col("n_purchases") / (F.col("tenure_days") / 30)) \
    .withColumn("estimated_monthly_value", F.col("avg_purchase_value") * F.col("purchase_frequency")) \
    .withColumn("estimated_annual_clv", F.col("estimated_monthly_value") * 12)

# Summary by strata
clv_summary = clv_analysis.groupBy("user_strata").agg(
    F.count("*").alias("customers"),
    F.mean("estimated_annual_clv").alias("avg_annual_clv"),
    F.sum("estimated_annual_clv").alias("total_annual_potential")
).orderBy(F.desc("avg_annual_clv"))

print("\n📊 Customer Lifetime Value by Segment:")
display(clv_summary)

# Save
clv_analysis.write.format("delta").mode("overwrite").saveAsTable("default.clv_analysis")
```

---

## 📊 Phase 4: Business Dashboard

### Step 4.1: Create Dashboard Metrics

```python
# Databricks Notebook: 04_Dashboard_Prep

# Summary metrics for dashboard
dashboard_metrics = spark.sql("""
SELECT 'Total Customers' as metric, COUNT(*) as value, 'count' as format
FROM default.user_summary

UNION ALL

SELECT 'Total Revenue ($)' as metric, SUM(total_spent) as value, 'currency' as format
FROM default.user_summary

UNION ALL

SELECT 'High Risk Customers' as metric, COUNT(*) as value, 'count' as format
FROM default.customer_risk_scores
WHERE risk_segment = 'High Risk'

UNION ALL

SELECT 'Revenue at Risk ($)' as metric, SUM(total_spent) as value, 'currency' as format
FROM default.customer_risk_scores
WHERE risk_segment = 'High Risk'

UNION ALL

SELECT 'Median Survival Time (days)' as metric, PERCENTILE(survival_time, 0.5) as value, 'days' as format
FROM default.survival_data

UNION ALL

SELECT 'Overall Churn Rate (%)' as metric, AVG(event) * 100 as value, 'percent' as format
FROM default.survival_data
""")

display(dashboard_metrics)
```

### Step 4.2: Create Dashboard (UI)

1. In Databricks, click **"Create"** → **"Dashboard"**
2. Name: `E-Commerce_Churn_Analysis_Dashboard`
3. Add Visualizations:

**KPI Cards:**
- Total Customers
- High-Risk Customers
- Revenue at Risk
- Median Customer Lifetime

**Charts:**
- Risk Segment Distribution (Pie Chart)
- Revenue by RFM Segment (Bar Chart)
- CLV by User Strata (Bar Chart)
- Risk Score Distribution (Histogram)

**Tables:**
- Top 10 High-Risk Customers
- RFM Segment Summary
- Cohort Performance

---

## 📁 Deliverables

### Databricks Notebooks:
1. ✅ `01_Load_Data.ipynb` - Upload & verify data
2. ✅ `02_Survival_Analysis.ipynb` - Cox model, risk scoring
3. ✅ `03_Data_Mining.ipynb` - RFM, cohort, CLV analysis
4. ✅ `04_Dashboard_Prep.ipynb` - Dashboard metrics

### Delta Tables Created:
- `default.user_summary` - 252K users with all features
- `default.survival_data` - Survival analysis dataset
- `default.customer_risk_scores` - Risk scores & time-to-churn
- `default.rfm_segments` - RFM customer segments
- `default.cohort_analysis` - Performance by user strata
- `default.clv_analysis` - Customer lifetime value

### Exported Files:
- `high_risk_interventions.csv` - Urgent intervention list

### Dashboard:
- `E-Commerce_Churn_Analysis_Dashboard` - Executive view

---

## 🎓 Course Module Coverage

| Module | Implementation | Location |
|--------|----------------|----------|
| **Survival Analysis** | Cox model, Kaplan-Meier, risk scoring | Notebook 02 |
| **Large-Scale Mining** | RFM (Spark SQL), cohort analysis, distributed processing | Notebook 03 |
| **Cloud Platform** | Databricks, Delta tables, distributed compute | All notebooks |

**Note:** Recommendation System implemented locally (requires event data not uploaded to Databricks)

---

## ⏱️ Execution Time

| Phase | Duration |
|-------|----------|
| Upload file | 5 min |
| Load into Spark | 5 min |
| Survival Analysis | 15-20 min |
| Data Mining | 10-15 min |
| Dashboard | 10 min |
| **Total** | **45-60 min** |

---

## ✅ Success Checklist

- [ ] File uploaded to DBFS
- [ ] Loaded into Delta table
- [ ] Survival model trained (C-index > 0.7)
- [ ] Risk scores calculated
- [ ] High-risk customers identified
- [ ] RFM segments created
- [ ] Dashboard published
- [ ] Results exported

---

## 🚀 Ready to Execute!

**Everything validated locally, ready for Databricks scale.**

*Execute notebooks sequentially for best results.*
