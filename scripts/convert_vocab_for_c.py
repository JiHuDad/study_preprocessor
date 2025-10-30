#!/usr/bin/env python3
"""
Python vocab.json을 C 엔진용 형식으로 변환하는 유틸리티

사용법:
    python convert_vocab_for_c.py <input_vocab.json> <output_vocab.json>

예시:
    python convert_vocab_for_c.py models/vocab.json models/vocab_c.json
"""

import json
import sys
from pathlib import Path


def convert_vocab_for_c_engine(vocab: dict) -> dict:
    """
    Python vocab을 C 엔진용으로 변환

    Python: {"template_string": index}
    C:      {"index": "template_string"}
    """
    # 이미 변환된 형식인지 확인
    sample_value = next(iter(vocab.values())) if vocab else None
    if sample_value and isinstance(sample_value, str) and len(sample_value) > 10:
        print("✅ 이미 C 엔진용 형식입니다")
        return vocab

    # 변환: {template: idx} → {str(idx): template}
    c_vocab = {str(idx): template for template, idx in vocab.items()}

    # 검증: 인덱스가 연속적인지 확인
    indices = sorted([int(k) for k in c_vocab.keys()])
    expected_indices = list(range(len(indices)))

    if indices != expected_indices:
        print(f"⚠️  경고: vocab 인덱스가 연속적이지 않습니다!")
        print(f"   기대: 0, 1, 2, ..., {len(indices)-1}")
        print(f"   실제: {indices[:10]}...")

    return c_vocab


def main():
    if len(sys.argv) < 2:
        print("사용법: python convert_vocab_for_c.py <input_vocab.json> [output_vocab.json]")
        print()
        print("예시:")
        print("  python convert_vocab_for_c.py models/vocab.json")
        print("  python convert_vocab_for_c.py models/vocab.json models/vocab_c.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # 출력 경로 결정
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        # 기본: 입력 파일과 같은 디렉토리에 vocab_c.json으로 저장
        output_path = input_path.parent / "vocab_c.json"

    # 입력 vocab 로드
    if not input_path.exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    print(f"📂 입력 파일: {input_path}")
    with open(input_path, 'r') as f:
        python_vocab = json.load(f)

    print(f"📊 Python vocab 로드: {len(python_vocab)} 템플릿")

    # 변환
    print("🔄 C 엔진용 형식으로 변환 중...")
    c_vocab = convert_vocab_for_c_engine(python_vocab)

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(c_vocab, f, indent=2, ensure_ascii=False)

    print(f"✅ C 엔진용 vocab 저장: {output_path}")
    print(f"📊 변환 완료: {len(c_vocab)} 템플릿")

    # 샘플 출력
    print("\n=== 변환 결과 샘플 (처음 5개) ===")
    for idx_str in sorted(c_vocab.keys(), key=lambda x: int(x))[:5]:
        template = c_vocab[idx_str]
        template_short = template[:60] + "..." if len(template) > 60 else template
        print(f"  {idx_str}: {template_short}")

    if len(c_vocab) > 5:
        print(f"  ... 및 {len(c_vocab) - 5}개 더")


if __name__ == "__main__":
    main()
