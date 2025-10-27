#!/usr/bin/env python3
"""
MapReduce Reducer: Aggregate total fare revenue by borough
Input: borough\tfare pairs from mapper (via stdin)
Output: Summary statistics per borough with timing information
"""
import sys
import time
from collections import defaultdict

def main():
    # Start timing
    start_time = time.time()
    sys.stderr.write(f"[REDUCER] Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Data structure to accumulate fares by borough
    borough_stats = defaultdict(lambda: {
        'total_revenue': 0.0,
        'trip_count': 0,
        'fares': []
    })
    
    # SHUFFLE & REDUCE PHASE
    read_start = time.time()
    sys.stderr.write("[REDUCER] Reading mapper output from stdin...\n")
    
    line_count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        try:
            # Parse mapper output
            borough, fare = line.split('\t')
            fare = float(fare)
            
            # Accumulate statistics
            borough_stats[borough]['total_revenue'] += fare
            borough_stats[borough]['trip_count'] += 1
            borough_stats[borough]['fares'].append(fare)
            
            line_count += 1
            
            # Progress indicator every 100k records
            if line_count % 100000 == 0:
                sys.stderr.write(f"[REDUCER] Processed {line_count:,} records...\n")
            
        except (ValueError, IndexError) as e:
            # Skip malformed lines
            continue
    
    read_time = time.time() - read_start
    sys.stderr.write(f"[REDUCER] Finished reading {line_count:,} records in {read_time:.2f} seconds\n")
    
    # Compute final statistics
    compute_start = time.time()
    sys.stderr.write("[REDUCER] Computing final statistics...\n")
    
    # Sort by total revenue (descending)
    sorted_boroughs = sorted(
        borough_stats.items(),
        key=lambda x: x[1]['total_revenue'],
        reverse=True
    )
    
    compute_time = time.time() - compute_start
    sys.stderr.write(f"[REDUCER] Statistics computed in {compute_time:.2f} seconds\n")
    
    # Output results
    output_start = time.time()
    
    print("\n" + "=" * 100)
    print(f"MapReduce Job Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print(f"\n{'Borough':<20} {'Total Revenue':<20} {'Trip Count':<15} {'Avg Fare':<12} {'Min Fare':<12} {'Max Fare':<12}")
    print("-" * 100)
    
    for borough, stats in sorted_boroughs:
        total_rev = stats['total_revenue']
        count = stats['trip_count']
        avg_fare = total_rev / count if count > 0 else 0
        min_fare = min(stats['fares']) if stats['fares'] else 0
        max_fare = max(stats['fares']) if stats['fares'] else 0
        
        print(f"{borough:<20} ${total_rev:>18,.2f} {count:>14,} ${avg_fare:>10.2f} ${min_fare:>10.2f} ${max_fare:>10.2f}")
    
    print("=" * 100)
    
    output_time = time.time() - output_start
    
    # Total time
    total_time = time.time() - start_time
    
    print(f"\n[PERFORMANCE METRICS]")
    print(f"  Total Records Processed: {line_count:,}")
    print(f"  Unique Boroughs: {len(borough_stats)}")
    print(f"  Read Time: {read_time:.2f} seconds")
    print(f"  Compute Time: {compute_time:.2f} seconds")
    print(f"  Output Time: {output_time:.2f} seconds")
    print(f"  Total Reducer Time: {total_time:.2f} seconds")
    print("=" * 100 + "\n")
    
    sys.stderr.write(f"[REDUCER] Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    sys.stderr.write(f"[REDUCER] Total execution time: {total_time:.2f} seconds\n")
    sys.stderr.write("=" * 70 + "\n")

if __name__ == "__main__":
    main()
