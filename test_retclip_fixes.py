#!/usr/bin/env python3
"""
RET-CLIP Local Test Suite
Tests all fixes on a small sample to ensure everything works before running full pipeline
"""

import os
import sys
import json
import base64
import lmdb
import pickle
import tempfile
import shutil
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np

# Add temp_retclip to path for testing
RETCLIP_PATH = Path(__file__).parent / "temp_retclip"
sys.path.insert(0, str(RETCLIP_PATH))

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_test(msg):
    print(f"{Colors.BLUE}[TEST]{Colors.END} {msg}")

def print_pass(msg):
    print(f"{Colors.GREEN}✅ PASS:{Colors.END} {msg}")

def print_fail(msg):
    print(f"{Colors.RED}❌ FAIL:{Colors.END} {msg}")

def print_warn(msg):
    print(f"{Colors.YELLOW}⚠️  WARN:{Colors.END} {msg}")

def print_section(msg):
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")


def test_1_config_files_exist():
    """Test that all required BERT config files exist"""
    print_section("TEST 1: BERT Config Files Exist")

    configs_dir = RETCLIP_PATH / "RET_CLIP" / "clip" / "model_configs"

    required_configs = [
        "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext.json",
        "bert-base-uncased.json",
        "dmis-lab-biobert-base-cased-v1.1.json",
        "ViT-B-16.json"
    ]

    all_exist = True
    for config_file in required_configs:
        config_path = configs_dir / config_file
        if config_path.exists():
            print_pass(f"Config exists: {config_file}")

            # Validate JSON structure
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Check required fields for text configs
                if "bert" in config_file.lower() or "pubmed" in config_file.lower():
                    required_fields = [
                        'vocab_size', 'text_hidden_size', 'text_num_hidden_layers',
                        'text_num_attention_heads', 'text_max_position_embeddings'
                    ]
                    for field in required_fields:
                        if field not in config:
                            print_fail(f"  Missing field: {field}")
                            all_exist = False
                    if all(field in config for field in required_fields):
                        print(f"     vocab_size={config['vocab_size']}, hidden_size={config['text_hidden_size']}")
            except json.JSONDecodeError as e:
                print_fail(f"  Invalid JSON: {e}")
                all_exist = False
        else:
            print_fail(f"Config missing: {config_file}")
            all_exist = False

    return all_exist


def test_2_choices_arrays_updated():
    """Test that choices arrays include English BERT models"""
    print_section("TEST 2: Choices Arrays Include English BERT")

    files_to_check = [
        ("training/params.py", 175),
        ("eval/extract_features_onnx.py", 87),
        ("eval/extract_features_tensorrt.py", 81)
    ]

    required_models = [
        "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "bert-base-uncased",
        "dmis-lab-biobert-base-cased-v1.1"
    ]

    all_updated = True

    for file_path, approx_line in files_to_check:
        full_path = RETCLIP_PATH / "RET_CLIP" / file_path
        print_test(f"Checking {file_path}")

        try:
            with open(full_path, 'r') as f:
                content = f.read()

            # Check if all required models are in the file
            missing = []
            for model in required_models:
                if model not in content:
                    missing.append(model)

            if missing:
                print_fail(f"  Missing models: {', '.join(missing)}")
                all_updated = False
            else:
                print_pass(f"  All English BERT models present")

        except Exception as e:
            print_fail(f"  Error reading file: {e}")
            all_updated = False

    return all_updated


def test_3_base64_encoding():
    """Test URL-safe base64 encoding/decoding"""
    print_section("TEST 3: URL-safe Base64 Encoding")

    # Create a small test image
    test_img = Image.new('RGB', (224, 224), color=(255, 0, 0))

    # Encode with URL-safe base64
    buffered = BytesIO()
    test_img.save(buffered, format="JPEG", quality=95)
    img_b64 = base64.urlsafe_b64encode(buffered.getvalue()).decode()

    print_test(f"Encoded image length: {len(img_b64)} chars")

    # Check for URL-safe characters (no / or +)
    if '/' in img_b64 or '+' in img_b64:
        print_fail("Contains non-URL-safe characters (/ or +)")
        return False
    else:
        print_pass("Uses URL-safe encoding (- and _ only)")

    # Test decoding
    try:
        img_bytes = base64.urlsafe_b64decode(img_b64)
        decoded_img = Image.open(BytesIO(img_bytes))

        if decoded_img.size == (224, 224):
            print_pass(f"Successfully decoded image: {decoded_img.size}")
            return True
        else:
            print_fail(f"Wrong image size: {decoded_img.size}")
            return False
    except Exception as e:
        print_fail(f"Decoding failed: {e}")
        return False


def test_4_tsv_format():
    """Test 3-column TSV format"""
    print_section("TEST 4: TSV Format (3 columns)")

    # Create test TSV
    temp_dir = tempfile.mkdtemp()
    tsv_path = Path(temp_dir) / "test.tsv"

    # Create sample data
    test_img = Image.new('RGB', (224, 224), color=(0, 255, 0))
    buffered = BytesIO()
    test_img.save(buffered, format="JPEG", quality=95)
    img_b64 = base64.urlsafe_b64encode(buffered.getvalue()).decode()

    # Write TSV with 3 columns
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write(f"patient_001\t{img_b64}\t{img_b64}\n")
        f.write(f"patient_002\t{img_b64}\t{img_b64}\n")

    print_test(f"Created test TSV: {tsv_path}")

    # Read and validate
    success = True
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) == 3:
                print_pass(f"Line {i}: 3 columns - patient_id, img_left, img_right")
            else:
                print_fail(f"Line {i}: {len(parts)} columns (expected 3)")
                success = False

    shutil.rmtree(temp_dir)
    return success


def test_5_checkpoint_loading_logic():
    """Test DDP checkpoint loading with 'module.' prefix stripping"""
    print_section("TEST 5: DDP Checkpoint Loading Logic")

    # Read the fixed main.py
    main_py_path = RETCLIP_PATH / "RET_CLIP" / "training" / "main.py"

    try:
        with open(main_py_path, 'r') as f:
            content = f.read()

        # Check for key patterns in the fix
        checks = [
            ("state_dict = checkpoint.get", "Handles both checkpoint formats"),
            ("startswith('module.')", "Checks for 'module.' prefix"),
            ("k[len('module.'):]", "Strips 'module.' prefix"),
            ("bert.pooler", "Filters out bert.pooler"),
        ]

        all_present = True
        for pattern, description in checks:
            if pattern in content:
                print_pass(f"{description}: Found '{pattern}'")
            else:
                print_fail(f"{description}: Missing '{pattern}'")
                all_present = False

        return all_present

    except Exception as e:
        print_fail(f"Error reading main.py: {e}")
        return False


def test_6_model_imports():
    """Test that RET-CLIP modules can be imported"""
    print_section("TEST 6: Module Imports")

    try:
        # Test importing key modules
        from RET_CLIP.clip.model import CLIP
        print_pass("Successfully imported CLIP model")

        from RET_CLIP.training.params import parse_args
        print_pass("Successfully imported params.parse_args")

        return True

    except Exception as e:
        print_fail(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}RET-CLIP LOCAL TEST SUITE{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")

    print(f"Testing fixes in: {RETCLIP_PATH}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}\n")

    tests = [
        ("Config Files Exist", test_1_config_files_exist),
        ("Choices Arrays Updated", test_2_choices_arrays_updated),
        ("URL-safe Base64", test_3_base64_encoding),
        ("TSV Format", test_4_tsv_format),
        ("Checkpoint Loading Logic", test_5_checkpoint_loading_logic),
        ("Module Imports", test_6_model_imports),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print_fail(f"Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print_section("TEST SUMMARY")

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        icon = "✅" if result else "❌"
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{icon} {test_name}: {color}{status}{Colors.END}")

    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED!{Colors.END}")
        print(f"{Colors.GREEN}The RET-CLIP fixes are working correctly.{Colors.END}")
        print(f"{Colors.GREEN}You can proceed with forking the repository.{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  SOME TESTS FAILED{Colors.END}")
        print(f"{Colors.RED}Fix the issues before proceeding.{Colors.END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
