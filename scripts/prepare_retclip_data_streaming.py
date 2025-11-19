#!/usr/bin/env python3
"""
RET-CLIP Data Preprocessing Pipeline with Streaming Support
Converts CSV with HuggingFace dataset indices to LMDB format for training

Uses streaming to avoid rate limits when downloading from HuggingFace.
Based on the approach from generate_retclip_prompts.ipynb
"""

import os
import sys
import argparse
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare RET-CLIP training data from HuggingFace with streaming")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV file with dataset_index, label, and prompt columns"
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="Peacein/color-fundus-eye",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data_prepared",
        help="Output directory for intermediate files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name (train, test)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Resize images to this size (default: 224)"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Save checkpoint every N samples"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between processing samples (to avoid rate limits)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if exists"
    )
    return parser.parse_args()


def authenticate_huggingface(token):
    """Authenticate with HuggingFace to avoid rate limits"""
    try:
        from huggingface_hub import login
        if token:
            login(token=token)
            print("✅ Authenticated with HuggingFace")
            return True
        else:
            print("⚠️  No HF_TOKEN provided - may encounter rate limits")
            return False
    except ImportError:
        print("Error: 'huggingface_hub' library not installed. Install with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"Warning: Could not authenticate with HuggingFace: {e}")
        print("Continuing without authentication - may encounter rate limits")
        return False


def image_to_base64(image, size=224):
    """Convert PIL Image to base64 string with resizing"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize image
    image = image.resize((size, size), Image.BICUBIC)

    # Convert to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_str = base64.urlsafe_b64encode(buffered.getvalue()).decode()
    return img_str


def load_huggingface_dataset_streaming(dataset_name, split):
    """Load dataset from HuggingFace in streaming mode"""
    try:
        from datasets import load_dataset
        print(f"Loading HuggingFace dataset in STREAMING mode: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        return dataset
    except ImportError:
        print("Error: 'datasets' library not installed. Install with: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def create_tsv_and_jsonl_streaming(args, df, hf_dataset_stream):
    """
    Create TSV and JSONL files using streaming approach

    TSV format: patient_id\timg_left_base64\timg_right_base64
    JSONL format: {"text_id": 0, "text": "prompt", "image_ids": ["patient_001"]}
    """
    os.makedirs(args.output_dir, exist_ok=True)

    tsv_path = os.path.join(args.output_dir, f"{args.split}_imgs.tsv")
    jsonl_path = os.path.join(args.output_dir, f"{args.split}_texts.jsonl")
    checkpoint_path = tsv_path.replace('.tsv', '_checkpoint.json')

    # Load checkpoint if resuming
    processed_indices = set()
    start_index = 0

    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            processed_indices = set(checkpoint.get('processed_indices', []))
            start_index = checkpoint.get('last_index', 0)
        print(f"✅ Resuming from checkpoint: {len(processed_indices)} samples already processed")

    # Open output files in append mode if resuming
    mode = 'a' if (args.resume and os.path.exists(tsv_path)) else 'w'
    tsv_file = open(tsv_path, mode, encoding='utf-8')
    jsonl_file = open(jsonl_path, mode, encoding='utf-8')

    # Track unique patients to avoid duplicates in TSV
    unique_patients_written = set()
    if args.resume and os.path.exists(tsv_path):
        # Read existing TSV to get patient IDs already written
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                patient_id = line.split('\t')[0]
                unique_patients_written.add(patient_id)

    print(f"\nProcessing {len(df)} samples from CSV...")
    print(f"Using streaming mode to avoid rate limits")
    print(f"Delay between samples: {args.delay}s")

    # Create iterator from streaming dataset
    dataset_iter = iter(hf_dataset_stream)

    # Build index mapping: we need to skip to the right position in stream
    # Since streaming doesn't support random access, we'll iterate through
    current_stream_index = 0

    # Process each row in CSV
    for csv_idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        dataset_index = int(row['dataset_index'])
        label = row['label']
        prompt = row['prompt']

        # Skip if already processed (for resume)
        if dataset_index in processed_indices:
            continue

        # Get patient ID
        patient_id = f"patient_{dataset_index:06d}"

        try:
            # Stream forward to the correct index
            while current_stream_index < dataset_index:
                try:
                    next(dataset_iter)
                    current_stream_index += 1
                except StopIteration:
                    print(f"\nWarning: Reached end of stream at index {current_stream_index}")
                    break

            # Get the sample at the correct index
            if current_stream_index == dataset_index:
                sample = next(dataset_iter)
                current_stream_index += 1
            else:
                print(f"Warning: Could not reach index {dataset_index}, skipping")
                continue

            # Extract image - handle single image (will use for both eyes)
            # Note: The dataset has 'image' field for single fundus image
            if 'image' in sample:
                image = sample['image']
                # For RET-CLIP dual eye architecture, use same image for both eyes
                # In production, you'd have separate left/right images
                left_b64 = image_to_base64(image, args.image_size)
                right_b64 = left_b64  # Using same image for both eyes
            else:
                print(f"Warning: No image field found for index {dataset_index}")
                continue

            # Write to TSV (only if patient not already written)
            if patient_id not in unique_patients_written:
                tsv_file.write(f"{patient_id}\t{left_b64}\t{right_b64}\n")
                unique_patients_written.add(patient_id)

            # Write to JSONL
            text_annotation = {
                "text_id": csv_idx,
                "text": prompt,
                "image_ids": [patient_id]
            }
            jsonl_file.write(json.dumps(text_annotation, ensure_ascii=False) + '\n')

            # Update checkpoint
            processed_indices.add(dataset_index)

            # Save checkpoint periodically
            if (csv_idx + 1) % args.checkpoint_interval == 0:
                with open(checkpoint_path, 'w') as f:
                    json.dump({
                        'processed_indices': list(processed_indices),
                        'last_index': dataset_index
                    }, f)
                tsv_file.flush()
                jsonl_file.flush()
                tqdm.write(f"✓ Checkpoint saved at sample {csv_idx + 1}")

            # Rate limiting delay
            if args.delay > 0:
                time.sleep(args.delay)

        except StopIteration:
            print(f"\nReached end of dataset stream at index {dataset_index}")
            break

        except Exception as e:
            print(f"\nError processing index {dataset_index}: {e}")
            # Save checkpoint on error
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'processed_indices': list(processed_indices),
                    'last_index': dataset_index
                }, f)
            tsv_file.flush()
            jsonl_file.flush()
            continue

    # Close files
    tsv_file.close()
    jsonl_file.close()

    # Clean up checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\n✓ Checkpoint removed after successful completion")

    print(f"\nSuccessfully created:")
    print(f"  - TSV with {len(unique_patients_written)} unique patients: {tsv_path}")
    print(f"  - JSONL with {len(processed_indices)} text-image pairs: {jsonl_path}")

    return tsv_path, jsonl_path


def main():
    args = parse_args()

    print("="*80)
    print("RET-CLIP Data Preparation Pipeline (STREAMING MODE)")
    print("="*80)

    # Get HuggingFace token from args or environment
    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Authenticate with HuggingFace
    print(f"\n1. Authenticating with HuggingFace")
    authenticate_huggingface(hf_token)

    # Load CSV file
    print(f"\n2. Loading CSV from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"   Loaded {len(df)} samples")
    print(f"   Columns: {df.columns.tolist()}")

    # Validate CSV structure
    required_columns = ['dataset_index', 'label', 'prompt']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: CSV must contain '{col}' column")
            sys.exit(1)

    # Load HuggingFace dataset in streaming mode
    print(f"\n3. Loading HuggingFace dataset in STREAMING mode: {args.hf_dataset_name}")
    hf_dataset_stream = load_huggingface_dataset_streaming(args.hf_dataset_name, args.split)

    # Create TSV and JSONL files with streaming
    print(f"\n4. Converting to TSV and JSONL format (streaming)")
    tsv_path, jsonl_path = create_tsv_and_jsonl_streaming(args, df, hf_dataset_stream)

    print("\n" + "="*80)
    print("Data preparation complete!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - TSV: {tsv_path}")
    print(f"  - JSONL: {jsonl_path}")
    print(f"\nNext steps:")
    print(f"  1. Run the LMDB builder script:")
    print(f"     python retclip/RET_CLIP/preprocess/build_lmdb_dataset_for_RET-CLIP.py \\")
    print(f"       --data_dir {args.output_dir} \\")
    print(f"       --splits {args.split} \\")
    print(f"       --lmdb_dir {args.output_dir}/lmdb")
    print(f"\n  2. Start training with the LMDB directory:")
    print(f"     {args.output_dir}/lmdb/{args.split}")


if __name__ == "__main__":
    main()
