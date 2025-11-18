#!/usr/bin/env python3
"""
Compare normalization results between Python and C implementations.

This helps identify if Python's mask_message and C's normalize_log_line
produce the same output for the same input.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anomaly_log_detector.preprocess import mask_message, PreprocessConfig


def test_normalization_samples():
    """Test normalization on various sample log lines"""

    test_cases = [
        "User root logged in from 192.168.1.100",
        "Connection failed: 0x7f8a3c001234",
        "Process pid=12345 started at /usr/bin/python3",
        "eth0: Link is up at 1000 Mbps",
        "Temperature: 45.67 degrees",
        "MAC address: 00:1B:44:11:3A:B7",
        "UUID: 550e8400-e29b-41d4-a716-446655440000",
        "Error at /home/user/file.txt line 123",
        "[  123.456789] Kernel message",
        "Mixed: /path/to/file pid=999 0xABCD 192.168.1.1 temp=36.5",
    ]

    print("=" * 80)
    print("PYTHON NORMALIZATION TEST")
    print("=" * 80)

    cfg = PreprocessConfig()  # Use default config

    for i, log_line in enumerate(test_cases, 1):
        normalized = mask_message(log_line, cfg)

        print(f"\n{i}. Original:")
        print(f"   {log_line}")
        print(f"   Normalized:")
        print(f"   {normalized}")

    print("\n" + "=" * 80)
    print("C NORMALIZATION TEST VECTORS")
    print("=" * 80)
    print("\nCopy these test cases to your C test file:\n")

    for i, log_line in enumerate(test_cases, 1):
        normalized = mask_message(log_line, cfg)

        print(f'// Test case {i}')
        print(f'test_normalize("{log_line}",')
        print(f'               "{normalized}");')
        print()


def compare_with_file(input_file: str, output_file: str = None):
    """
    Normalize log lines from input file and save to output file.

    You can then run the same file through C implementation and compare.
    """
    print(f"Reading log lines from: {input_file}")

    cfg = PreprocessConfig()

    with open(input_file, 'r') as f:
        lines = f.readlines()

    normalized_lines = []
    for line in lines:
        line = line.rstrip('\n')
        if line:
            normalized = mask_message(line, cfg)
            normalized_lines.append(normalized)

    if output_file:
        with open(output_file, 'w') as f:
            for norm in normalized_lines:
                f.write(norm + '\n')
        print(f"✅ Normalized {len(normalized_lines)} lines to: {output_file}")
    else:
        print("\n=== Normalized Output ===")
        for i, (orig, norm) in enumerate(zip(lines, normalized_lines), 1):
            print(f"\n{i}. {orig.rstrip()}")
            print(f"   → {norm}")


def main():
    if len(sys.argv) == 1:
        # No arguments: show test samples
        test_normalization_samples()
    elif len(sys.argv) >= 2:
        # Arguments provided: process file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None

        if not Path(input_file).exists():
            print(f"ERROR: File not found: {input_file}")
            sys.exit(1)

        compare_with_file(input_file, output_file)


if __name__ == "__main__":
    main()
