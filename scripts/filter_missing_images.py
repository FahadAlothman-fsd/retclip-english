#!/usr/bin/env python3
"""
Filter JSONL files to only keep entries that have corresponding images in TSV
"""

import json
import sys

def filter_jsonl(tsv_path, jsonl_path, output_path):
    """Remove JSONL entries that don't have matching images in TSV"""

    # Read all patient IDs from TSV
    print(f"Reading patient IDs from {tsv_path}...")
    patient_ids = set()
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            patient_id = line.split('\t')[0]
            patient_ids.add(patient_id)

    print(f"Found {len(patient_ids)} patients with images")

    # Filter JSONL to only keep entries with matching patient IDs
    print(f"Filtering {jsonl_path}...")
    kept = 0
    removed = 0

    with open(jsonl_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            obj = json.loads(line.strip())
            # Check if all image_ids exist in TSV
            if all(pid in patient_ids for pid in obj['image_ids']):
                fout.write(line)
                kept += 1
            else:
                removed += 1

    print(f"\n✅ Filtering complete!")
    print(f"   Kept: {kept} entries")
    print(f"   Removed: {removed} entries")
    print(f"   Output: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python filter_missing_images.py <tsv_path> <jsonl_path> <output_path>")
        sys.exit(1)

    filter_jsonl(sys.argv[1], sys.argv[2], sys.argv[3])
