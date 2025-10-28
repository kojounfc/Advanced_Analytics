#!/usr/bin/env python3
"""
Lightweight MapReduce Mapper
Reads CSV from stdin, emits borough-fare pairs
"""
import sys
import time

start_time = time.time()
sys.stderr.write(f"[MAPPER] Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

count = 0
for line in sys.stdin:
    try:
        line = line.strip()
        if not line:
            continue
            
        # Parse CSV: borough,total_amount
        parts = line.split(',')
        if len(parts) >= 2:
            borough = parts[0] if parts[0] else 'Unknown'
            total_amount = float(parts[1])
            
            if total_amount > 0:
                # Emit key-value pair
                print(f"{borough}\t{total_amount}")
                count += 1
                
                # Progress indicator
                if count % 100000 == 0:
                    sys.stderr.write(f"[MAPPER] Processed {count:,} records\n")
                    
    except (ValueError, IndexError):
        # Skip malformed lines
        continue

elapsed = time.time() - start_time
sys.stderr.write(f"[MAPPER] Completed: {count:,} records in {elapsed:.2f} seconds\n")
sys.stderr.write("="*70 + "\n")
