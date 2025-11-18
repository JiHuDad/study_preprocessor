#!/usr/bin/env python3
"""
Diagnose vocabulary and ONNX model mismatch issues.

This tool helps identify:
1. Vocab size mismatch between vocab.json and ONNX model
2. Template ID mapping issues
3. Normalization differences between Python and C
"""

import json
import sys
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed. Install with: pip install onnxruntime")
    sys.exit(1)


def check_vocab_json(vocab_path: str) -> dict:
    """Load and validate vocab.json"""
    print(f"\n=== Checking vocab.json: {vocab_path} ===")

    if not Path(vocab_path).exists():
        print(f"❌ File not found: {vocab_path}")
        return {}

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    print(f"✅ Vocab loaded: {len(vocab)} templates")

    # Detect vocab format
    sample_key = next(iter(vocab.keys())) if vocab else None
    sample_value = next(iter(vocab.values())) if vocab else None

    # Determine format:
    # C format: {"0": "template_string", ...} - keys are numeric strings, values are templates
    # Python format: {"template_string": 0, ...} - keys are templates, values are numeric
    is_c_format = False
    is_python_format = False

    if sample_key and sample_value:
        try:
            int(sample_key)
            is_c_format = isinstance(sample_value, str)
        except ValueError:
            is_python_format = isinstance(sample_value, int)

    if is_c_format:
        print("📋 Format: C format (index → template)")

        # Check if indices are consecutive and zero-based
        try:
            indices = sorted([int(k) for k in vocab.keys()])
        except ValueError as e:
            print(f"❌ ERROR: Cannot parse vocab keys as integers: {e}")
            print(f"   Sample key: {sample_key}")
            return vocab

        expected_indices = list(range(len(indices)))

        if indices != expected_indices:
            print(f"❌ ERROR: Non-consecutive or non-zero-based indices!")
            print(f"   Expected: 0, 1, 2, ..., {len(indices)-1}")
            print(f"   Actual: {indices[:20]}...")

            missing = set(expected_indices) - set(indices)
            if missing:
                print(f"   Missing indices: {sorted(missing)[:20]}...")

            extra = set(indices) - set(expected_indices)
            if extra:
                print(f"   Extra indices: {sorted(extra)[:20]}...")
        else:
            print(f"✅ Indices are consecutive: 0 to {len(indices)-1}")

        # Show first and last few templates
        print("\n=== First 5 templates ===")
        for idx in sorted(indices)[:5]:
            template = vocab[str(idx)]
            template_short = template[:80] if len(template) > 80 else template
            print(f"  {idx}: {template_short}")

        if len(indices) > 5:
            print(f"\n=== Last 5 templates ===")
            for idx in sorted(indices)[-5:]:
                template = vocab[str(idx)]
                template_short = template[:80] if len(template) > 80 else template
                print(f"  {idx}: {template_short}")

    elif is_python_format:
        print("📋 Format: Python format (template → index)")
        print("⚠️  WARNING: This is Python training format, not C inference format!")
        print("   C engine expects: {\"0\": \"template\", \"1\": \"template\", ...}")
        print("   But received: {\"template\": 0, \"template\": 1, ...}")
        print("\n💡 Convert using: python scripts/convert_vocab_for_c.py")

        # Show index distribution
        indices = sorted(vocab.values())
        print(f"\n=== Index range: {min(indices)} to {max(indices)} ({len(indices)} templates) ===")

        # Show first 5 templates
        print("\n=== First 5 templates (by index) ===")
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for template, idx in sorted_vocab[:5]:
            template_short = template[:80] if len(template) > 80 else template
            print(f"  {idx}: {template_short}")

    else:
        print("⚠️  Unknown vocab format")
        print(f"   Sample: {sample_key} → {sample_value}")

    return vocab


def check_onnx_model(model_path: str) -> dict:
    """Load and inspect ONNX model"""
    print(f"\n=== Checking ONNX model: {model_path} ===")

    if not Path(model_path).exists():
        print(f"❌ File not found: {model_path}")
        return {}

    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        return {}

    print(f"✅ ONNX model loaded")

    # Get input info
    print("\n=== Model Inputs ===")
    for inp in session.get_inputs():
        print(f"  Name: {inp.name}")
        print(f"  Shape: {inp.shape}")
        print(f"  Type: {inp.type}")

    # Get output info
    print("\n=== Model Outputs ===")
    for out in session.get_outputs():
        print(f"  Name: {out.name}")
        print(f"  Shape: {out.shape}")
        print(f"  Type: {out.type}")

    # Extract vocab_size from output shape
    output_shape = session.get_outputs()[0].shape
    vocab_size = None

    if len(output_shape) >= 2:
        # Typically output shape is [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
        if len(output_shape) == 2:
            vocab_size = output_shape[1] if isinstance(output_shape[1], int) else None
        elif len(output_shape) == 3:
            vocab_size = output_shape[2] if isinstance(output_shape[2], int) else None

    if vocab_size:
        print(f"\n✅ Detected vocab_size from output shape: {vocab_size}")
    else:
        print(f"\n⚠️  Could not detect vocab_size from output shape: {output_shape}")

    return {
        'session': session,
        'vocab_size': vocab_size,
        'input_shape': session.get_inputs()[0].shape,
        'output_shape': output_shape
    }


def compare_vocab_and_model(vocab: dict, model_info: dict):
    """Compare vocab.json and ONNX model"""
    print("\n=== Comparing vocab.json and ONNX model ===")

    if not vocab or not model_info:
        print("❌ Cannot compare: missing vocab or model info")
        return

    vocab_size_json = len(vocab)
    vocab_size_onnx = model_info.get('vocab_size')

    if vocab_size_onnx is None:
        print("⚠️  Cannot determine vocab_size from ONNX model")
        return

    print(f"  vocab.json size: {vocab_size_json}")
    print(f"  ONNX vocab_size: {vocab_size_onnx}")

    if vocab_size_json == vocab_size_onnx:
        print("✅ Vocab sizes match!")
    else:
        print(f"❌ ERROR: Vocab size mismatch!")
        print(f"   Difference: {abs(vocab_size_json - vocab_size_onnx)}")

        if vocab_size_json > vocab_size_onnx:
            print(f"   vocab.json has {vocab_size_json - vocab_size_onnx} extra templates")
            print(f"   ONNX model expects indices 0-{vocab_size_onnx-1}")
        else:
            print(f"   vocab.json is missing {vocab_size_onnx - vocab_size_json} templates")
            print(f"   Model expects {vocab_size_onnx} templates but only {vocab_size_json} provided")


def test_inference(model_info: dict, vocab: dict, seq_len: int = 10):
    """Test a simple inference to check if it works"""
    print(f"\n=== Testing Inference (seq_len={seq_len}) ===")

    if not model_info or 'session' not in model_info:
        print("❌ Cannot test: model not loaded")
        return

    session = model_info['session']
    vocab_size = len(vocab) if vocab else 100

    # Create dummy input
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randint(0, min(vocab_size, 10), size=(1, seq_len), dtype=np.int64)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Input sample: {dummy_input[0][:5]}...")

    try:
        output = session.run(None, {input_name: dummy_input})
        print(f"✅ Inference successful!")
        print(f"  Output shape: {output[0].shape}")
        print(f"  Output sample (first 5 logits): {output[0][0][:5]}")

        # Check if output looks reasonable
        if np.any(np.isnan(output[0])):
            print("⚠️  WARNING: Output contains NaN values!")
        if np.any(np.isinf(output[0])):
            print("⚠️  WARNING: Output contains Inf values!")

    except Exception as e:
        print(f"❌ Inference failed: {e}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python diagnose_vocab_mismatch.py <vocab.json> <model.onnx> [seq_len]")
        print("\nExample:")
        print("  python diagnose_vocab_mismatch.py hybrid_system/inference/models/vocab.json hybrid_system/inference/models/deeplog.onnx 10")
        sys.exit(1)

    vocab_path = sys.argv[1]
    model_path = sys.argv[2]
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    print("=" * 80)
    print("VOCAB AND ONNX MODEL DIAGNOSTIC TOOL")
    print("=" * 80)

    # Check vocab.json
    vocab = check_vocab_json(vocab_path)

    # Check ONNX model
    model_info = check_onnx_model(model_path)

    # Compare
    compare_vocab_and_model(vocab, model_info)

    # Test inference
    test_inference(model_info, vocab, seq_len)

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
