#!/usr/bin/env python3
"""
Export vocabulary with actual template strings for C inference engine.

This script reads parsed.parquet and extracts the mapping from template_id to
actual template strings, which is required by the C inference engine.
"""

import json
import sys
from pathlib import Path
import pandas as pd


def export_vocab_with_templates(parsed_parquet: str, output_json: str) -> None:
    """
    Create vocabulary JSON file with template strings.

    Args:
        parsed_parquet: Path to parsed.parquet file
        output_json: Path to output vocab.json file

    Format:
        {
            "0": "User <*> logged in from <*>",
            "1": "Connection failed: <*>",
            ...
        }
    """
    print(f"Reading parsed data from: {parsed_parquet}")
    df = pd.read_parquet(parsed_parquet)

    # Check required columns
    if 'template_id' not in df.columns:
        print("ERROR: 'template_id' column not found in parquet file")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Try to find template string column
    template_col = None
    for col_name in ['template', 'template_str', 'log_template', 'EventTemplate']:
        if col_name in df.columns:
            template_col = col_name
            break

    if template_col is None:
        print("ERROR: No template string column found in parquet file")
        print(f"Available columns: {df.columns.tolist()}")
        print("\nSearched for: 'template', 'template_str', 'log_template', 'EventTemplate'")
        sys.exit(1)

    print(f"Using template column: {template_col}")
    print(f"Total rows: {len(df)}")

    # Build mapping: template_id -> template_string
    # Take the first occurrence of each template_id
    template_map = {}

    for _, row in df[['template_id', template_col]].drop_duplicates('template_id').iterrows():
        template_id = str(row['template_id'])
        template_str = str(row[template_col])

        # Skip NaN values
        if pd.isna(template_id) or pd.isna(template_str):
            continue

        template_map[template_id] = template_str

    print(f"Extracted {len(template_map)} unique templates")

    # Show first few templates
    print("\nFirst 5 templates:")
    for i, (tid, tpl) in enumerate(sorted(template_map.items(), key=lambda x: int(x[0]))[:5]):
        print(f"  {tid}: {tpl[:80]}")

    # Save to JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template_map, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Vocabulary exported to: {output_json}")
    print(f"   Total templates: {len(template_map)}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python export_vocab_with_templates.py <parsed.parquet> <output_vocab.json>")
        print("\nExample:")
        print("  python export_vocab_with_templates.py data/processed/parsed.parquet hybrid_system/inference/models/vocab.json")
        sys.exit(1)

    parsed_parquet = sys.argv[1]
    output_json = sys.argv[2]

    if not Path(parsed_parquet).exists():
        print(f"ERROR: Input file not found: {parsed_parquet}")
        sys.exit(1)

    export_vocab_with_templates(parsed_parquet, output_json)


if __name__ == "__main__":
    main()
