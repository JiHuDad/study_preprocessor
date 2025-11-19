#!/usr/bin/env python3
"""
Train DeepLog without Drain3 - using regex masking only.

This ensures Python training and C inference use identical template extraction.
"""

import pandas as pd
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anomaly_log_detector.preprocess import mask_message, PreprocessConfig, parse_line


def preprocess_without_drain3(input_log: str, output_parquet: str):
    """
    Preprocess logs using ONLY regex masking (no Drain3).

    This creates templates identical to C inference engine.
    """
    config = PreprocessConfig(
        mask_dates=True,  # CRITICAL: Must match C config
        mask_paths=True,
        mask_hex=True,
        mask_ips=True,
        mask_mac=True,
        mask_uuid=True,
        mask_pid_fields=True,
        mask_device_numbers=True,
        mask_numbers=True,
    )

    rows = []

    with open(input_log, 'r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            ts, host, proc, msg = parse_line(line)

            # Apply ONLY regex masking (same as C)
            template = mask_message(msg, config)

            rows.append({
                'line_no': idx,
                'timestamp': ts,
                'host': host,
                'process': proc,
                'raw': msg,
                'template': template,  # This is now identical to C normalization
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['timestamp', 'line_no'], kind='stable', na_position='first')

    # Assign template indices (sorted order for consistency)
    unique_templates = sorted(df['template'].dropna().unique())
    template_to_idx = {tpl: idx for idx, tpl in enumerate(unique_templates)}

    df['template_index'] = df['template'].map(template_to_idx)

    # Save
    df.to_parquet(output_parquet, index=False)

    print(f"✅ Processed {len(df)} log lines")
    print(f"✅ Found {len(unique_templates)} unique templates (regex-only)")
    print(f"✅ Saved to {output_parquet}")

    # Save vocab
    vocab_path = Path(output_parquet).parent / "vocab.json"
    import json

    # C format: {"0": "template", "1": "template", ...}
    vocab_c = {str(idx): tpl for tpl, idx in template_to_idx.items()}

    with open(vocab_path, 'w') as f:
        json.dump(vocab_c, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved vocab to {vocab_path}")
    print(f"\n=== Sample templates (first 10) ===")
    for i, tpl in enumerate(unique_templates[:10]):
        print(f"  {i}: {tpl[:80]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_without_drain3.py <input.log> <output.parquet>")
        print("\nExample:")
        print("  python scripts/train_without_drain3.py data/raw/hdfs.log data/processed/parsed_regex_only.parquet")
        sys.exit(1)

    input_log = sys.argv[1]
    output_parquet = sys.argv[2]

    preprocess_without_drain3(input_log, output_parquet)
