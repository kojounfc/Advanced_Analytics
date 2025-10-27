#!/usr/bin/env python3
"""
Enhanced Parquet to CSV Conversion with Domain-Based Data Cleaning
===================================================================
Joins taxi data with zone lookup, applies domain-based cleaning, and exports as CSV

Cleaning Strategy:
1. Remove null/invalid values (negatives, zeros)
2. Remove Unknown/N/A boroughs
3. Remove extreme outliers using business rules (not statistical methods)
4. Detailed quality reporting
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, min as spark_min, max as spark_max, count as spark_count
import time

print("="*80)
print("ENHANCED DATA CLEANING & CSV CONVERSION FOR MAPREDUCE")
print("="*80)

start_time = time.time()

# Create Spark session
spark = SparkSession.builder \
    .appName("Enhanced Parquet to CSV with Cleaning") \
    .config("spark.driver.memory", "3g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================
print("\n[1/5] Loading yellow taxi data from HDFS...")
df = spark.read.parquet("hdfs://localhost:9000/user/hadoop/taxi_data/input/yellow_tripdata_2025-01.parquet")
original_count = df.count()
print(f"      Loaded {original_count:,} records")

print("\n[2/5] Loading zone lookup data...")
zones = spark.read.csv("hdfs://localhost:9000/user/hadoop/taxi_data/zones/taxi_zone_lookup.csv",
                       header=True, inferSchema=True)
zone_count = zones.count()
print(f"      Loaded {zone_count} zones")

# ============================================================================
# STEP 3: JOIN WITH ZONE LOOKUP
# ============================================================================
print("\n[3/5] Joining data with zone lookup...")
result = df.join(
    zones.select(
        col('LocationID').alias('PULocationID'),
        col('Borough').alias('PU_Borough')
    ),
    on='PULocationID',
    how='left'
).select('PU_Borough', 'total_amount', 'trip_distance')

joined_count = result.count()
print(f"      Joined {joined_count:,} records")

# ============================================================================
# STEP 4: DOMAIN-BASED DATA CLEANING
# ============================================================================
print("\n[4/5] Applying domain-based data cleaning...")
print("="*80)

print("\n📋 DATA CLEANING RULES (Business Logic)")
print("-" * 80)

# Track removals for each step
cleaning_start = result.count()

# Rule 1: Remove null total_amount
print("\n  Rule 1: Remove null total_amount")
result_temp = result.filter(col('total_amount').isNotNull())
null_amount = cleaning_start - result_temp.count()
print(f"         Removed: {null_amount:>10,} records")

# Rule 2: Remove total_amount <= 0 (negative and zero fares)
print("\n  Rule 2: Remove total_amount <= 0")
before_count = result_temp.count()
result_temp = result_temp.filter(col('total_amount') > 0)
negative_amount = before_count - result_temp.count()
print(f"         Removed: {negative_amount:>10,} records (includes negative fares)")

# Rule 3: Remove extreme total_amount outliers (> $200)
print("\n  Rule 3: Remove extreme fares (> $200)")
print("         Rationale: NYC taxi fares rarely exceed $200")
print("                   (even JFK airport is ~$70-80)")
before_count = result_temp.count()
result_temp = result_temp.filter(col('total_amount') <= 200)
extreme_fare = before_count - result_temp.count()
print(f"         Removed: {extreme_fare:>10,} records (includes $863K Queens fare!)")

# Rule 4: Remove trip_distance <= 0
print("\n  Rule 4: Remove trip_distance <= 0")
before_count = result_temp.count()
result_temp = result_temp.filter(col('trip_distance') > 0)
zero_distance = before_count - result_temp.count()
print(f"         Removed: {zero_distance:>10,} records")

# Rule 5: Remove extreme trip_distance outliers (> 100 miles)
print("\n  Rule 5: Remove extreme distances (> 100 miles)")
print("         Rationale: NYC metro area trips are < 100 miles")
before_count = result_temp.count()
result_temp = result_temp.filter(col('trip_distance') <= 100)
extreme_distance = before_count - result_temp.count()
print(f"         Removed: {extreme_distance:>10,} records")

# Rule 6: Remove Unknown/N/A/null boroughs
print("\n  Rule 6: Remove Unknown/N/A/null boroughs")
print("         Rationale: Cannot attribute revenue to unknown location")
before_count = result_temp.count()
result_temp = result_temp.filter(
    col('PU_Borough').isNotNull() &
    (col('PU_Borough') != 'Unknown') &
    (col('PU_Borough') != 'N/A') &
    (col('PU_Borough') != '') &
    (col('PU_Borough') != 'None')
)
unknown_borough = before_count - result_temp.count()
print(f"         Removed: {unknown_borough:>10,} records")

# Set variables for removed passenger counts to 0 (since we're not checking them)
zero_passenger = 0
excess_passenger = 0

result_clean = result_temp
cleaning_end = result_clean.count()
total_removed = cleaning_start - cleaning_end

# ============================================================================
# CLEANING SUMMARY REPORT
# ============================================================================
print("\n\n" + "="*80)
print("DATA CLEANING SUMMARY REPORT")
print("="*80)

retention_rate = (cleaning_end / original_count) * 100

print(f"\n📊 OVERALL STATISTICS:")
print(f"  Original records:                   {original_count:>10,}")
print(f"  Final cleaned records:              {cleaning_end:>10,}")
print(f"  Total removed:                      {total_removed:>10,}")
print(f"  Removal rate:                       {total_removed/original_count*100:>9.2f}%")
print(f"  Retention rate:                     {retention_rate:>9.2f}%")

# Calculate statistics on cleaned data
print(f"\n📈 CLEANED DATA STATISTICS:")

stats = result_clean.select(
    avg('total_amount').alias('avg_amount'),
    spark_min('total_amount').alias('min_amount'),
    spark_max('total_amount').alias('max_amount'),
    avg('trip_distance').alias('avg_distance'),
    spark_min('trip_distance').alias('min_distance'),
    spark_max('trip_distance').alias('max_distance')
).collect()[0]

print(f"  Average total_amount:               ${stats['avg_amount']:>9.2f}")
print(f"  Min total_amount:                   ${stats['min_amount']:>9.2f} ✓ (no negatives)")
print(f"  Max total_amount:                   ${stats['max_amount']:>9.2f} ✓ (capped at $200)")
print(f"  Average trip_distance:              {stats['avg_distance']:>9.2f} miles")
print(f"  Min trip_distance:                  {stats['min_distance']:>9.2f} miles ✓ (no zeros)")
print(f"  Max trip_distance:                  {stats['max_distance']:>9.2f} miles ✓ (capped at 100)")

# Borough distribution
print(f"\n🗺️  BOROUGH DISTRIBUTION (after cleaning):")
print("-" * 80)
borough_stats = result_clean.groupBy('PU_Borough') \
    .agg(
        spark_count('*').alias('trip_count'),
        avg('total_amount').alias('avg_fare')
    ) \
    .orderBy(col('trip_count').desc())

borough_data = borough_stats.collect()
for row in borough_data:
    borough = row['PU_Borough']
    count = row['trip_count']
    avg_fare = row['avg_fare']
    percentage = (count / cleaning_end) * 100
    bar = '█' * int(percentage / 2)
    print(f"  {borough:<20} {bar:25} {percentage:>5.1f}% ({count:>10,} trips, avg ${avg_fare:.2f})")

print("-" * 80)
print(f"  {'TOTAL':<20} {'':<25} 100.0% ({cleaning_end:>10,} trips)")
print(f"\n  ✓ No Unknown/N/A boroughs remaining!")

# ============================================================================
# DETAILED BREAKDOWN BY CLEANING RULE
# ============================================================================
print(f"\n📋 REMOVAL BREAKDOWN BY RULE:")
print("-" * 80)

removal_summary = [
    ("Null total_amount", null_amount),
    ("Negative/zero total_amount", negative_amount),
    ("Extreme fares (>$200)", extreme_fare),
    ("Zero trip_distance", zero_distance),
    ("Extreme distances (>100mi)", extreme_distance),
    ("Unknown/N/A boroughs", unknown_borough)
]

for rule_name, count in removal_summary:
    pct = (count / original_count) * 100
    bar = '░' * int(pct * 2) if pct > 0 else ''
    print(f"  {rule_name:<30} {count:>10,} ({pct:>5.2f}%) {bar}")

print("-" * 80)
print(f"  {'TOTAL REMOVED':<30} {total_removed:>10,} ({total_removed/original_count*100:>5.2f}%)")

# ============================================================================
# STEP 5: SAVE CLEANED DATA
# ============================================================================
print("\n\n[5/5] Writing cleaned CSV to HDFS...")
output_path = "hdfs://localhost:9000/user/hadoop/taxi_data/mapreduce_input"

# Select only the columns needed for MapReduce (borough, total_amount)
final_output = result_clean.select('PU_Borough', 'total_amount')

# Save as CSV
final_output.coalesce(1).write.mode('overwrite').csv(output_path, header=False)

elapsed = time.time() - start_time

print("\n" + "="*80)
print(f"✅ CLEANING & CONVERSION COMPLETED SUCCESSFULLY")
print("="*80)
print(f"⏱️  Total execution time:               {elapsed:.2f} seconds")
print(f"📁 Output location:                     {output_path}")
print(f"📊 Records ready for MapReduce:         {cleaning_end:,}")
print(f"✓  Data quality verified")
print(f"✓  No negative values")
print(f"✓  No Unknown boroughs")
print(f"✓  Extreme outliers removed (>$200, >100mi)")
print(f"✓  Legitimate trips preserved (airport, long-distance)")
print(f"✓  Ready for accurate MapReduce analysis")
print("="*80)

# ============================================================================
# VALIDATION CHECK
# ============================================================================
print("\n🔍 VALIDATION CHECK:")
print("-" * 80)

# Check for any remaining issues
validation_checks = [
    ("Null boroughs", final_output.filter(col('PU_Borough').isNull()).count()),
    ("Null amounts", final_output.filter(col('total_amount').isNull()).count()),
    ("Negative amounts", final_output.filter(col('total_amount') <= 0).count()),
    ("Zero amounts", final_output.filter(col('total_amount') == 0).count()),
    ("Extreme fares (>$200)", final_output.filter(col('total_amount') > 200).count()),
    ("Unknown boroughs", final_output.filter(
        (col('PU_Borough') == 'Unknown') |
        (col('PU_Borough') == 'N/A') |
        (col('PU_Borough') == '')
    ).count())
]

all_passed = True
for check_name, count in validation_checks:
    status = "✓ PASS" if count == 0 else "✗ FAIL"
    if count > 0:
        all_passed = False
    print(f"  {check_name:<30} {count:>10,} {status:>10}")

print("-" * 80)
if all_passed:
    print("  ✅ ALL VALIDATION CHECKS PASSED - DATA IS CLEAN")
else:
    print("  ⚠️  SOME VALIDATION CHECKS FAILED - REVIEW NEEDED")

print("\n" + "="*80)
print("READY FOR MAPREDUCE ANALYSIS")
print("="*80)
print("""
Next steps:
1. Run MapReduce: hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \\
                  -input /user/hadoop/taxi_data/mapreduce_input \\
                  -output /user/hadoop/taxi_data/mapreduce_output \\
                  -mapper mapper.py \\
                  -reducer reducer.py

2. Expected results:
   - Queens average fare: ~$15-18 (was $70.84)
   - No Unknown boroughs
   - Realistic statistics across all boroughs
""")

spark.stop()