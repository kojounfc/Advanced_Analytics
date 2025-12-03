# E-Commerce Customer Churn Analysis - Implementation Plan

**Course:** DAMO630 - Advanced Data Analytics
**Project Type:** E-commerce Customer Retention & Recommendation
**Platform:** Databricks (Mandatory Cloud-Based Implementation)
**Dataset:** REES46 E-commerce Events (Oct-Nov 2019, ~110M events)

---

## 📋 Executive Summary

### Business Problem
**How can we predict and prevent customer churn in e-commerce to maximize customer lifetime value?**

### Objective
Build an end-to-end analytics pipeline that:
1. Identifies customers at risk of churning
2. Predicts time-to-churn for intervention planning
3. Recommends personalized products to increase engagement
4. Processes large-scale data efficiently on cloud infrastructure

### Course Modules Integrated (3 Required)
✅ **1. Survival Analysis** - Predict time-to-churn, identify risk factors
✅ **2. Recommendation Systems** - Personalized product recommendations
✅ **3. Mining Large-Scale Datasets** - Hadoop/Spark on Databricks cloud

---

## 🎯 Business Value Proposition

### Key Stakeholders
- **Marketing Team:** Target at-risk customers with retention campaigns
- **Product Team:** Understand which products drive retention
- **Executive Team:** ROI on retention investments, CLV optimization

### Expected Outcomes
1. **Churn Prediction Model:** Identify 80%+ of churners before they leave
2. **Intervention Timing:** Optimal window for retention campaigns
3. **Personalized Recommendations:** Increase engagement by 20-30%
4. **Cost Savings:** 5x cheaper to retain than acquire new customers

### Success Metrics
- **Model Performance:** C-index > 0.75 for survival model, RMSE < 1.0 for recommendations
- **Business Impact:** 15-25% reduction in churn rate
- **Scalability:** Process 100M+ events in < 10 minutes on Databricks

---

## 📊 Data Architecture

### Current Status
✅ **Sampling Complete:** 252,899 users sampled (10% of 2.5M total)
✅ **Purchase Retention:** 95.73% (145K of 151K purchases retained)
✅ **Buyer Retention:** 95.19% (115K of 121K buyers retained)
✅ **Data Reduction:** 81% (10M → 1.9M events)

### Data Pipeline Flow
```
Raw CSV Files (110M events)
    ↓
[Local Sampling - COMPLETE]
    ↓
Sampled User IDs (252K users)
    ↓
[Upload to Databricks]
    ↓
Filter & Load to Spark (1.9M events)
    ↓
Feature Engineering (User-level metrics)
    ↓
[Module 1: Survival Analysis] → Churn Predictions
[Module 2: Recommendation System] → Product Recommendations
[Module 3: Large-Scale Mining] → Behavioral Patterns
    ↓
Business Insights & Visualizations
```

---

## 🔧 Technical Implementation Plan

### PHASE 1: Data Upload & Cloud Setup (Week 1)
**Status:** NEXT STEP

#### Tasks:
1. **Upload to Databricks**
   - Create Databricks workspace (Community Edition or paid)
   - Upload `sampled_user_ids.csv` to DBFS
   - Upload Oct/Nov CSV files to DBFS (or use original and filter)
   - Create tables in Databricks catalog

2. **Data Filtering in Databricks**
   ```python
   # Load sampled user IDs
   sampled_ids = spark.read.csv("dbfs:/sampled_user_ids.csv", header=True)

   # Load and filter original data
   oct_data = spark.read.csv("dbfs:/2019-Oct.csv", header=True)
   nov_data = spark.read.csv("dbfs:/2019-Nov.csv", header=True)

   combined = oct_data.union(nov_data)
   filtered = combined.join(sampled_ids, "user_id", "inner")

   # Save as Delta table for performance
   filtered.write.format("delta").saveAsTable("ecommerce_events_sampled")
   ```

3. **Validation**
   - Verify event counts match local sampling (~1.9M events)
   - Verify purchase counts (~145K purchases)
   - Check data quality (nulls, duplicates, outliers)

**Deliverable:** Databricks notebook `01_Data_Upload_Setup.ipynb`

---

### PHASE 2: Feature Engineering (Week 1-2)

#### Customer-Level Features (Survival Analysis)
```python
# RFM Features
- recency: Days since last purchase
- frequency: Number of purchases in period
- monetary: Total revenue generated

# Engagement Features
- session_count: Number of sessions
- avg_session_duration: Average time per session
- products_viewed: Unique products browsed
- cart_adds: Items added to cart
- cart_abandonment_rate: (carts - purchases) / carts

# Temporal Features
- first_purchase_date: When customer first bought
- last_purchase_date: When customer last bought
- tenure_days: Days active on platform
- purchase_frequency: Purchases per week
- days_between_purchases: Average gap

# Behavioral Features
- view_to_cart_rate: Cart adds / Views
- cart_to_purchase_rate: Purchases / Cart adds
- avg_order_value: Revenue / Purchases
- favorite_category: Most browsed category
- brand_diversity: Number of unique brands
```

#### Event-Level Features (Recommendation System)
```python
# User-Item Interaction Matrix
- user_id: Customer identifier
- product_id: Product identifier
- interaction_type: view=1, cart=2, purchase=3 (implicit feedback)
- interaction_count: Number of interactions
- last_interaction_date: Recency

# Product Features
- product_category: Category hierarchy
- product_brand: Brand name
- avg_price: Average price point
- popularity_score: How many users interacted
```

**Deliverable:** Databricks notebook `02_Feature_Engineering.ipynb`

---

### PHASE 3: Module 1 - Survival Analysis (Week 2)

#### Objective
Predict **time-to-churn** and identify **risk factors** for customer attrition.

#### Definition of Churn
- **Event:** Customer stops making purchases (no purchase in 30+ days)
- **Time:** Days from first purchase to last purchase (or censoring)
- **Censoring:** Right-censored if customer still active at observation end

#### Survival Analysis Workflow

**Step 1: Create Survival Dataset**
```python
from datetime import datetime, timedelta

def create_survival_dataset(user_summary):
    """
    Create survival analysis dataset with time-to-event and censoring.
    """
    observation_end = datetime(2019, 11, 30)  # End of November
    churn_threshold_days = 30

    survival_data = user_summary.copy()

    # Calculate time-to-event
    survival_data['last_purchase'] = pd.to_datetime(survival_data['last_purchase'])
    survival_data['first_purchase'] = pd.to_datetime(survival_data['first_purchase'])

    # Survival time (in days)
    survival_data['survival_time'] = (survival_data['last_purchase'] -
                                       survival_data['first_purchase']).dt.days + 1

    # Event indicator (1 = churned, 0 = censored)
    days_since_last = (observation_end - survival_data['last_purchase']).dt.days
    survival_data['event'] = (days_since_last > churn_threshold_days).astype(int)

    return survival_data[survival_data['n_purchases'] > 0]  # Only buyers
```

**Step 2: Kaplan-Meier Survival Curves**
```python
from lifelines import KaplanMeierFitter

# Overall survival curve
kmf = KaplanMeierFitter()
kmf.fit(survival_data['survival_time'], survival_data['event'])
kmf.plot_survival_function()
plt.title('Customer Survival Curve: Probability of Remaining Active')
plt.xlabel('Days Since First Purchase')
plt.ylabel('Survival Probability')
plt.show()

# By user strata
for strata in ['power_buyer', 'high_value_buyer', 'buyer']:
    data = survival_data[survival_data['user_strata'] == strata]
    kmf.fit(data['survival_time'], data['event'], label=strata)
    kmf.plot_survival_function()
```

**Step 3: Cox Proportional Hazards Model**
```python
from lifelines import CoxPHFitter

# Select features
features = [
    'recency', 'frequency', 'monetary',
    'avg_order_value', 'cart_abandonment_rate',
    'view_to_cart_rate', 'session_count',
    'products_viewed', 'brand_diversity'
]

# Prepare data
cox_data = survival_data[features + ['survival_time', 'event']].dropna()

# Fit Cox model
cph = CoxPHFitter()
cph.fit(cox_data, duration_col='survival_time', event_col='event')

# Display results
cph.print_summary()
cph.plot()  # Hazard ratios with confidence intervals
```

**Step 4: Risk Scoring & Segmentation**
```python
# Predict risk scores (higher = more likely to churn)
survival_data['risk_score'] = cph.predict_partial_hazard(cox_data[features])

# Segment customers by risk
survival_data['risk_segment'] = pd.cut(
    survival_data['risk_score'],
    bins=[0, 0.33, 0.66, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Business insights
high_risk = survival_data[survival_data['risk_segment'] == 'High Risk']
print(f"High-risk customers: {len(high_risk):,}")
print(f"Revenue at risk: ${high_risk['monetary'].sum():,.2f}")
```

**Step 5: Time-to-Churn Predictions**
```python
# Predict survival probability at specific time points
time_points = [7, 14, 30, 60, 90]  # Days
survival_probs = cph.predict_survival_function(cox_data[features], times=time_points)

# Identify customers likely to churn in next 30 days
prob_30d = survival_probs.loc[30]
churn_next_30 = survival_data[prob_30d < 0.5]  # <50% chance of surviving
```

#### Expected Outputs
- Kaplan-Meier curves by customer segment
- Cox model coefficients (which features increase churn risk)
- Risk scores for all customers
- List of high-risk customers for intervention
- Predicted time-to-churn for each customer

**Deliverable:** Databricks notebook `03_Survival_Analysis.ipynb`

---

### PHASE 4: Module 2 - Recommendation System (Week 3)

#### Objective
Build personalized product recommendations to increase engagement and reduce churn.

#### Recommendation Strategy
**Collaborative Filtering** - Find similar users and recommend products they purchased

#### Implementation: ALS (Alternating Least Squares)

**Step 1: Create User-Item Interaction Matrix**
```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Create implicit feedback (views=1, cart=2, purchase=3)
interactions = spark.sql("""
    SELECT
        user_id,
        product_id,
        CASE
            WHEN event_type = 'purchase' THEN 3
            WHEN event_type = 'cart' THEN 2
            WHEN event_type = 'view' THEN 1
        END as rating
    FROM ecommerce_events_sampled
""")

# Aggregate multiple interactions
interactions_agg = interactions.groupBy("user_id", "product_id") \
    .agg(F.max("rating").alias("rating"))
```

**Step 2: Train ALS Model**
```python
# Create index mappings
from pyspark.ml.feature import StringIndexer

user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx")
product_indexer = StringIndexer(inputCol="product_id", outputCol="product_idx")

interactions_indexed = user_indexer.fit(interactions_agg).transform(interactions_agg)
interactions_indexed = product_indexer.fit(interactions_indexed).transform(interactions_indexed)

# Train/test split
train, test = interactions_indexed.randomSplit([0.8, 0.2], seed=42)

# Configure ALS
als = ALS(
    userCol="user_idx",
    itemCol="product_idx",
    ratingCol="rating",
    rank=50,              # Number of latent factors
    maxIter=15,           # Iterations
    regParam=0.01,        # Regularization
    implicitPrefs=True,   # Implicit feedback
    coldStartStrategy="drop",
    seed=42
)

# Train model
model = als.fit(train)

# Evaluate
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse:.3f}")
```

**Step 3: Generate Recommendations**
```python
# Top 10 products for each user
user_recs = model.recommendForAllUsers(10)

# Top 10 users for each product (for marketing)
product_recs = model.recommendForAllItems(10)

# Example: Recommendations for high-risk customers
high_risk_users = survival_data[survival_data['risk_segment'] == 'High Risk']['user_id']
high_risk_recs = user_recs.filter(F.col("user_id").isin(high_risk_users.tolist()))
```

**Step 4: Evaluation Metrics**
```python
# Precision@K, Recall@K for top-10 recommendations
def precision_at_k(predictions, k=10):
    # Users who actually purchased recommended items
    pass

# Coverage: % of catalog recommended
# Diversity: Variety of categories recommended
# Novelty: Are we recommending popular vs. niche items?
```

**Step 5: Business Integration**
```python
# Anti-churn recommendation strategy
def anti_churn_recommendations(user_id, model, survival_data):
    """
    Personalized recommendations for at-risk customers.
    Focus on products similar to past purchases to re-engage.
    """
    user_risk = survival_data[survival_data['user_id'] == user_id]['risk_segment'].values[0]

    if user_risk == 'High Risk':
        # Recommend familiar products (same category/brand)
        recs = get_recommendations(user_id, model, filter_by='familiarity')
    else:
        # Recommend discovery items
        recs = get_recommendations(user_id, model, filter_by='novelty')

    return recs
```

#### Expected Outputs
- Trained ALS recommendation model
- Top-10 product recommendations for each user
- Evaluation metrics (RMSE, Precision@K, Coverage)
- Personalized recommendations for high-risk customers
- Product affinity matrix (similar products)

**Deliverable:** Databricks notebook `04_Recommendation_System.ipynb`

---

### PHASE 5: Module 3 - Large-Scale Data Mining (Week 3-4)

#### Objective
Demonstrate scalability and extract behavioral patterns using Spark distributed processing.

#### Mining Tasks

**Task 1: Customer Segmentation (RFM Analysis)**
```python
# Distributed RFM calculation using Spark
from pyspark.sql.window import Window

rfm = spark.sql("""
    WITH purchase_events AS (
        SELECT user_id, event_time, price
        FROM ecommerce_events_sampled
        WHERE event_type = 'purchase'
    ),
    rfm_metrics AS (
        SELECT
            user_id,
            DATEDIFF('2019-11-30', MAX(event_time)) as recency,
            COUNT(*) as frequency,
            SUM(price) as monetary
        FROM purchase_events
        GROUP BY user_id
    )
    SELECT
        user_id,
        recency,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency DESC) as R_score,
        NTILE(5) OVER (ORDER BY frequency ASC) as F_score,
        NTILE(5) OVER (ORDER BY monetary ASC) as M_score
    FROM rfm_metrics
""")

# Create RFM segments
rfm = rfm.withColumn("RFM_segment", F.concat(F.col("R_score"), F.col("F_score"), F.col("M_score")))
```

**Task 2: Product Association Rules (Market Basket Analysis)**
```python
from pyspark.ml.fpm import FPGrowth

# Create baskets (products purchased together in sessions)
baskets = spark.sql("""
    SELECT user_session, COLLECT_LIST(product_id) as items
    FROM ecommerce_events_sampled
    WHERE event_type = 'purchase'
    GROUP BY user_session
    HAVING COUNT(*) > 1
""")

# FP-Growth for frequent itemsets
fp = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.3)
model = fp.fit(baskets)

# Frequent patterns
model.freqItemsets.show(20)

# Association rules (If bought X, likely to buy Y)
model.associationRules.show(20)
```

**Task 3: Conversion Funnel Analysis**
```python
# Multi-stage funnel: View → Cart → Purchase
funnel = spark.sql("""
    WITH user_funnel AS (
        SELECT
            user_id,
            MAX(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) as viewed,
            MAX(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) as added_cart,
            MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchased
        FROM ecommerce_events_sampled
        GROUP BY user_id
    )
    SELECT
        SUM(viewed) as total_viewers,
        SUM(added_cart) as reached_cart,
        SUM(purchased) as completed_purchase,
        SUM(added_cart) / SUM(viewed) as view_to_cart_rate,
        SUM(purchased) / SUM(added_cart) as cart_to_purchase_rate,
        SUM(purchased) / SUM(viewed) as overall_conversion_rate
    FROM user_funnel
""")
```

**Task 4: Time-Series Analysis (Daily Trends)**
```python
# Daily metrics using Spark aggregations
daily_metrics = spark.sql("""
    SELECT
        DATE(event_time) as date,
        COUNT(*) as total_events,
        COUNT(DISTINCT user_id) as active_users,
        SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) as daily_revenue,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases
    FROM ecommerce_events_sampled
    GROUP BY DATE(event_time)
    ORDER BY date
""")
```

**Task 5: Performance Benchmarking**
```python
# Demonstrate scalability
import time

def benchmark_query(query, description):
    start = time.time()
    result = spark.sql(query).collect()
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.2f}s")
    return result

# Compare performance with/without optimization
# - Partitioning strategies
# - Caching strategies
# - Broadcast joins
```

#### Expected Outputs
- RFM customer segments
- Association rules (product bundles)
- Conversion funnel metrics
- Daily/weekly trend analysis
- Performance benchmarks (query times, scalability charts)

**Deliverable:** Databricks notebook `05_Large_Scale_Mining.ipynb`

---

### PHASE 6: Business Insights & Visualizations (Week 4)

#### Objective
Create compelling, business-oriented visualizations for executive presentation.

#### Key Visualizations

**1. Churn Risk Dashboard**
- Risk score distribution (histogram)
- Survival curves by segment (line chart)
- Revenue at risk by segment (bar chart)
- Geographic/demographic churn patterns (heatmap)

**2. Recommendation Performance**
- Precision/Recall curves
- Coverage analysis (what % of catalog is recommended)
- Diversity metrics (category distribution)
- A/B test simulation (predicted engagement lift)

**3. Customer Segmentation**
- RFM 3D scatter plot (R, F, M axes)
- Segment profitability matrix
- Customer lifetime value by segment
- Migration paths (how customers move between segments)

**4. Product Insights**
- Association rules network graph
- Category performance (revenue, conversion rate)
- Product affinity heatmap
- Trending vs. declining products

**5. Operational Metrics**
- Daily active users trend
- Conversion funnel (Sankey diagram)
- Session duration distribution
- Cart abandonment rate over time

#### Visualization Tools
- **Databricks Notebooks:** Built-in display() for quick charts
- **Matplotlib/Seaborn:** Custom Python visualizations
- **Plotly:** Interactive dashboards
- **Databricks Dashboards:** Shareable executive dashboards

**Deliverable:** Databricks notebook `06_Business_Visualizations.ipynb` + Dashboard

---

### PHASE 7: Model Deployment & Productionization (Week 4)

#### Real-Time Churn Scoring API
```python
# Save trained models
cph_model.save("dbfs:/models/cox_survival_model.pkl")
als_model.write().overwrite().save("dbfs:/models/als_recommendation_model")

# Create scoring function
def score_customer(user_id):
    """
    Real-time churn risk scoring and recommendations.
    """
    # Get user features
    user_features = get_user_features(user_id)

    # Predict churn risk
    risk_score = cph_model.predict_partial_hazard(user_features)
    survival_prob_30d = cph_model.predict_survival_function(user_features, times=[30])

    # Get recommendations
    recs = als_model.recommendForUsers(user_id, numItems=10)

    return {
        'user_id': user_id,
        'risk_score': float(risk_score),
        'churn_probability_30d': 1 - float(survival_prob_30d),
        'risk_segment': 'High' if risk_score > 0.66 else 'Medium' if risk_score > 0.33 else 'Low',
        'recommended_products': recs,
        'intervention_urgency': 'Immediate' if survival_prob_30d < 0.5 else 'Monitor'
    }

# Batch scoring for all customers
batch_scores = spark.createDataFrame([score_customer(uid) for uid in user_ids])
batch_scores.write.format("delta").mode("overwrite").saveAsTable("customer_churn_scores")
```

#### Databricks Job Scheduling
- **Daily:** Update customer risk scores
- **Weekly:** Retrain recommendation model
- **Monthly:** Retrain survival model with new data

**Deliverable:** Databricks notebook `07_Model_Deployment.ipynb` + Scheduled Jobs

---

## 📈 Business Recommendations

### Immediate Actions (Week 1-2)
1. **High-Risk Customer Intervention**
   - Send personalized retention offers to top 10% risk customers
   - Offer free shipping/discounts on recommended products
   - Targeted email campaigns with product suggestions

2. **Cart Abandonment Recovery**
   - Automated email reminders for abandoned carts
   - Retarget with recommended similar products
   - Time-sensitive discount codes

### Medium-Term Strategies (Month 1-3)
1. **Loyalty Program Design**
   - Reward frequent purchasers (power buyers)
   - Gamification to increase engagement
   - Tier-based benefits (bronze/silver/gold)

2. **Product Bundling**
   - Use association rules to create bundles
   - Cross-sell complementary products
   - Category-based promotions

### Long-Term Initiatives (Quarter 1-2)
1. **Predictive Inventory Management**
   - Forecast demand based on survival models
   - Stock recommended products proactively
   - Optimize warehouse placement

2. **Customer Lifetime Value Optimization**
   - Acquisition cost vs. predicted CLV analysis
   - Segment-specific marketing budgets
   - Churn prevention ROI tracking

---

## 📚 Deliverables Summary

### Code Deliverables
1. ✅ `ADA-Final_Project_Sampling_Local.ipynb` - Local sampling (COMPLETE)
2. `01_Data_Upload_Setup.ipynb` - Databricks data loading
3. `02_Feature_Engineering.ipynb` - Customer/product features
4. `03_Survival_Analysis.ipynb` - Churn prediction models
5. `04_Recommendation_System.ipynb` - ALS recommender
6. `05_Large_Scale_Mining.ipynb` - Spark distributed processing
7. `06_Business_Visualizations.ipynb` - Executive dashboards
8. `07_Model_Deployment.ipynb` - Production deployment

### Report Deliverables
1. **Technical Report** (20-30 pages)
   - Introduction & problem statement
   - Literature review (survival analysis, RecSys, big data)
   - Methodology (detailed algorithms)
   - Results & evaluation metrics
   - Limitations & future work
   - Appendices (code snippets, full results)

2. **Business Presentation** (15-20 slides)
   - Executive summary
   - Business problem & value proposition
   - Key findings (3-5 insights)
   - Recommendations (actionable items)
   - ROI projections
   - Implementation roadmap

3. **Code Documentation**
   - README.md with setup instructions
   - Requirements.txt / environment.yml
   - Data dictionary
   - API documentation (if deployed)

---

## ⏱️ Timeline & Milestones

| Week | Phase | Deliverables | Status |
|------|-------|-------------|--------|
| **Week 1** | Sampling | Local sampling notebook | ✅ COMPLETE |
| **Week 1** | Cloud Setup | Upload data to Databricks | 🔄 NEXT |
| **Week 2** | Feature Engineering | Customer & product features | ⏳ Pending |
| **Week 2** | Survival Analysis | Cox model, risk scores | ⏳ Pending |
| **Week 3** | Recommendations | ALS model, product recs | ⏳ Pending |
| **Week 3** | Large-Scale Mining | RFM, association rules | ⏳ Pending |
| **Week 4** | Visualizations | Dashboards & insights | ⏳ Pending |
| **Week 4** | Deployment | Production scoring pipeline | ⏳ Pending |
| **Week 5** | Documentation | Technical report | ⏳ Pending |
| **Week 5** | Presentation | Business slides | ⏳ Pending |

---

## 🎓 Rubric Alignment

### Problem Description & Business Relevance (90-100 Target)
✅ Clear business problem: Customer churn costs 5x more than retention
✅ Quantifiable impact: 15-25% churn reduction = $X revenue saved
✅ Stakeholder alignment: Marketing, Product, Executive teams

### Integration of Course Modules (90-100 Target)
✅ **Survival Analysis:** Time-to-churn prediction (core methodology)
✅ **Recommendation Systems:** Personalized engagement (retention driver)
✅ **Large-Scale Mining:** Spark processing on Databricks (scalability proof)
✅ Seamless integration: Survival informs recommendations (high-risk users get targeted recs)

### Cloud-Based Implementation (90-100 Target)
✅ Full Databricks pipeline (data, models, deployment)
✅ Delta Lake for storage
✅ Spark for distributed processing
✅ Scheduled jobs for production scoring

### Analytics Quality & Rigor (90-100 Target)
✅ Proper survival analysis (right-censoring, Cox PH assumptions)
✅ Train/test splits, cross-validation
✅ Multiple evaluation metrics (C-index, RMSE, Precision@K)
✅ Statistical significance testing

### Business-Oriented Visualizations (90-100 Target)
✅ Executive dashboards (Databricks Dashboards)
✅ Clear storytelling: Problem → Analysis → Recommendations
✅ Actionable insights highlighted
✅ ROI projections included

### Presentation & Documentation (90-100 Target)
✅ Fully reproducible notebooks (seed=42, documented code)
✅ Professional documentation (README, data dictionary)
✅ Business presentation tailored to executives
✅ Technical report with academic rigor

---

## 🚀 Next Steps (Immediate Actions)

### Step 1: Set Up Databricks Workspace
1. Go to [Databricks Community Edition](https://community.cloud.databricks.com) or use paid account
2. Create new workspace: "DAMO630_Final_Project"
3. Create cluster: Runtime 13.3 LTS (Spark 3.4.1, Scala 2.12)
4. Install libraries: `lifelines`, `implicit` (if not pre-installed)

### Step 2: Upload Data
1. Upload `sampled_user_ids.csv` to DBFS: `/FileStore/final_project/`
2. Upload `2019-Oct.csv` and `2019-Nov.csv` (or filter locally and upload sampled events)
3. Create notebook: `01_Data_Upload_Setup.ipynb`

### Step 3: Start Feature Engineering
1. Create notebook: `02_Feature_Engineering.ipynb`
2. Load data into Spark DataFrame
3. Build RFM features, engagement metrics, temporal features
4. Save as Delta table: `customer_features`

### Step 4: Begin Survival Analysis
1. Create notebook: `03_Survival_Analysis.ipynb`
2. Install `lifelines`: `%pip install lifelines`
3. Implement Kaplan-Meier curves
4. Fit Cox Proportional Hazards model

---

## 📞 Support & Resources

### Documentation
- [Databricks Survival Analysis Guide](https://docs.databricks.com)
- [Lifelines Documentation](https://lifelines.readthedocs.io)
- [Spark MLlib ALS](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)

### Code References
- Survival Analysis: Use `lifelines` library (Python)
- Recommendations: Use `pyspark.ml.recommendation.ALS`
- Visualizations: Use `matplotlib`, `seaborn`, `plotly`

### Timeline Check-ins
- **Week 2 Check:** Is survival model trained?
- **Week 3 Check:** Are recommendations generated?
- **Week 4 Check:** Are visualizations complete?

---

## 🎯 Success Criteria

### Technical Success
✅ Survival model C-index > 0.75
✅ Recommendation RMSE < 1.0
✅ Process 1.9M events in < 5 minutes
✅ All notebooks run end-to-end without errors

### Business Success
✅ Identify 10,000+ high-risk customers
✅ Estimate revenue at risk
✅ Provide 10 product recommendations per user
✅ Demonstrate 15-25% potential churn reduction

### Academic Success
✅ All 3 modules seamlessly integrated
✅ Cloud implementation complete
✅ Professional documentation
✅ Compelling business presentation
✅ Target grade: 90-100%

---

**END OF IMPLEMENTATION PLAN**

*This plan is designed to be executed incrementally. Start with Phase 1 (Cloud Setup) and proceed sequentially. Each phase builds on the previous one.*

*Estimated total implementation time: 4-5 weeks with 15-20 hours per week.*

*Good luck! 🚀*
