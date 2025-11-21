#!/usr/bin/env python3
"""
Simple validation script to check if all RET-CLIP fixes were applied correctly
No dependencies required - just reads files and checks for expected patterns
"""

import json
from pathlib import Path

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
BOLD = '\033[1m'
END = '\033[0m'

def check(condition, message):
    """Print check result"""
    if condition:
        print(f"{GREEN}✅{END} {message}")
        return True
    else:
        print(f"{RED}❌{END} {message}")
        return False

def main():
    print(f"\n{BOLD}{'='*80}{END}")
    print(f"{BOLD}RET-CLIP Fixes Validation{END}")
    print(f"{BOLD}{'='*80}{END}\n")

    retclip_path = Path("temp_retclip/RET_CLIP")
    results = []

    # Check 1: BERT config files exist
    print(f"\n{BOLD}1. BERT Config Files{END}")
    configs_dir = retclip_path / "clip" / "model_configs"

    for config_name in [
        "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext.json",
        "bert-base-uncased.json",
        "dmis-lab-biobert-base-cased-v1.1.json"
    ]:
        config_path = configs_dir / config_name
        result = check(config_path.exists(), f"{config_name} exists")
        results.append(result)

        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                result = check('vocab_size' in config, f"  → Has vocab_size: {config.get('vocab_size')}")
                results.append(result)
            except:
                results.append(False)

    # Check 2: params.py updated
    print(f"\n{BOLD}2. training/params.py{END}")
    params_path = retclip_path / "training" / "params.py"
    with open(params_path) as f:
        params_content = f.read()

    for model in [
        "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "bert-base-uncased",
        "dmis-lab-biobert-base-cased-v1.1"
    ]:
        result = check(model in params_content, f"Contains {model}")
        results.append(result)

    # Check 3: extract_features_onnx.py updated
    print(f"\n{BOLD}3. eval/extract_features_onnx.py{END}")
    onnx_path = retclip_path / "eval" / "extract_features_onnx.py"
    with open(onnx_path) as f:
        onnx_content = f.read()

    result = check("bert-base-uncased" in onnx_content, "Contains bert-base-uncased")
    results.append(result)

    # Check 4: extract_features_tensorrt.py updated
    print(f"\n{BOLD}4. eval/extract_features_tensorrt.py{END}")
    tensorrt_path = retclip_path / "eval" / "extract_features_tensorrt.py"
    with open(tensorrt_path) as f:
        tensorrt_content = f.read()

    result = check("bert-base-uncased" in tensorrt_content, "Contains bert-base-uncased")
    results.append(result)

    # Check 5: main.py DDP fix
    print(f"\n{BOLD}5. training/main.py (DDP Fix){END}")
    main_path = retclip_path / "training" / "main.py"
    with open(main_path) as f:
        main_content = f.read()

    patterns = [
        ("checkpoint.get(\"state_dict\"", "Handles both checkpoint formats"),
        ("startswith('module.')", "Checks for module. prefix"),
        ("k[len('module.'):]", "Strips module. prefix"),
    ]

    for pattern, description in patterns:
        result = check(pattern in main_content, description)
        results.append(result)

    # Summary
    print(f"\n{BOLD}{'='*80}{END}")
    print(f"{BOLD}SUMMARY{END}")
    print(f"{BOLD}{'='*80}{END}\n")

    passed = sum(results)
    total = len(results)

    print(f"Checks passed: {passed}/{total}")

    if passed == total:
        print(f"\n{GREEN}{BOLD}🎉 ALL FIXES APPLIED CORRECTLY!{END}")
        print(f"{GREEN}You can now:{END}")
        print(f"{GREEN}  1. Fork the original RET-CLIP repo on GitHub{END}")
        print(f"{GREEN}  2. Copy these fixes to your fork{END}")
        print(f"{GREEN}  3. Push to GitHub{END}")
        print(f"{GREEN}  4. Update the notebook to clone from your fork{END}")
        return 0
    else:
        print(f"\n{RED}{BOLD}⚠️ SOME CHECKS FAILED{END}")
        print(f"{RED}Please review the failures above{END}")
        return 1

if __name__ == "__main__":
    exit(main())
