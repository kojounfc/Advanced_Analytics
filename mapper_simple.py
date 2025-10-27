#!/usr/bin/env python3
"""
Lightweight Mapper: Read from stdin, emit borough-fare pairs
This version assumes data is pre-processed and fed via stdin
"""
import sys
import csv
import time

start_time = time.time()
sys.stderr.write(f"[MAPPER] Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

count = 0
for line in sys.stdin:
    try:
        # Skip header
        if line.startswith('VendorID') or line.startswith('borough'):
            continue
            
        # Parse CSV format: borough,total_amount
        parts = line.strip().split(',')
        if len(parts) >= 2:
            borough = parts[0]
            total_amount = float(parts[1])
            
            if total_amount > 0:
                print(f"{borough}\t{total_amount}")
                count += 1
                
                if count % 100000 == 0:
                    sys.stderr.write(f"[MAPPER] Processed {count:,} records\n")
    except (ValueError, IndexError):
        continue

elapsed = time.time() - start_time
sys.stderr.write(f"[MAPPER] Completed {count:,} records in {elapsed:.2f} seconds\n")
