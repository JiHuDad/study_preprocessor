#!/bin/bash
# Enhanced DeepLog Demo
# 알림 폭주 방지 기능이 적용된 DeepLog inference 데모

set -e

DEMO_DIR="demo_enhanced_deeplog_output"
echo "======================================"
echo "Enhanced DeepLog Demo"
echo "======================================"
echo ""

# 정리 및 디렉토리 생성
echo "🧹 데모 디렉토리 준비..."
rm -rf "$DEMO_DIR"
mkdir -p "$DEMO_DIR"

# 1. 합성 로그 생성
echo "1️⃣  합성 로그 생성 (이상 포함)..."
study-preprocess gen-synth \
  --out "$DEMO_DIR/demo.log" \
  --lines 2000 \
  --anomaly-rate 0.05

echo "✅ 합성 로그 생성 완료"
echo ""

# 2. 전처리
echo "2️⃣  로그 전처리 (Drain3 템플릿 마이닝)..."
study-preprocess parse \
  --input "$DEMO_DIR/demo.log" \
  --out-dir "$DEMO_DIR/processed" \
  --drain-state "$DEMO_DIR/drain3_state.json"

echo "✅ 전처리 완료"
echo ""

# 3. DeepLog 입력 생성
echo "3️⃣  DeepLog 입력 생성 (시퀀스 + vocab)..."
study-preprocess build-deeplog \
  --parsed "$DEMO_DIR/processed/parsed.parquet" \
  --out-dir "$DEMO_DIR/processed"

echo "✅ DeepLog 입력 생성 완료"
echo ""

# 4. DeepLog 모델 학습
echo "4️⃣  DeepLog 모델 학습..."
study-preprocess deeplog-train \
  --seq "$DEMO_DIR/processed/sequences.parquet" \
  --vocab "$DEMO_DIR/processed/vocab.json" \
  --out "$DEMO_DIR/deeplog_model.pth" \
  --seq-len 20 \
  --epochs 3

echo "✅ DeepLog 학습 완료"
echo ""

# 5a. 기본 DeepLog 추론 (비교용)
echo "5️⃣ a. 기본 DeepLog 추론 (기존 방식)..."
study-preprocess deeplog-infer \
  --seq "$DEMO_DIR/processed/sequences.parquet" \
  --model "$DEMO_DIR/deeplog_model.pth" \
  --k 3

echo "✅ 기본 추론 완료"
echo ""

# 5b. Enhanced DeepLog 추론 (Top-K)
echo "5️⃣ b. Enhanced DeepLog 추론 (Top-K, K-of-N, Cooldown)..."
study-preprocess deeplog-infer-enhanced \
  --seq "$DEMO_DIR/processed/sequences.parquet" \
  --parsed "$DEMO_DIR/processed/parsed.parquet" \
  --model "$DEMO_DIR/deeplog_model.pth" \
  --vocab "$DEMO_DIR/processed/vocab.json" \
  --top-k 3 \
  --k-of-n-k 7 \
  --k-of-n-n 10 \
  --cooldown-seq 60 \
  --cooldown-novelty 60 \
  --session-timeout 300 \
  --entity-column host \
  --out-dir "$DEMO_DIR/enhanced_topk"

echo "✅ Enhanced 추론 (Top-K) 완료"
echo ""

# 5c. Enhanced DeepLog 추론 (Top-P)
echo "5️⃣ c. Enhanced DeepLog 추론 (Top-P)..."
study-preprocess deeplog-infer-enhanced \
  --seq "$DEMO_DIR/processed/sequences.parquet" \
  --parsed "$DEMO_DIR/processed/parsed.parquet" \
  --model "$DEMO_DIR/deeplog_model.pth" \
  --vocab "$DEMO_DIR/processed/vocab.json" \
  --top-p 0.9 \
  --k-of-n-k 7 \
  --k-of-n-n 10 \
  --entity-column host \
  --out-dir "$DEMO_DIR/enhanced_topp"

echo "✅ Enhanced 추론 (Top-P) 완료"
echo ""

# 6. 결과 비교
echo "6️⃣  결과 비교..."
echo ""

# 기본 DeepLog 결과
echo "📊 기본 DeepLog 결과:"
python3 << 'EOF'
import pandas as pd
import sys

try:
    df = pd.read_parquet('demo_enhanced_deeplog_output/processed/deeplog_infer.parquet')
    total = len(df)
    violations = len(df[df['in_topk'] == False])
    rate = violations / total if total > 0 else 0

    print(f"  총 시퀀스: {total:,}개")
    print(f"  위반 (알림): {violations:,}개")
    print(f"  위반율: {rate:.1%}")
    print()
except Exception as e:
    print(f"  ❌ 파일 읽기 실패: {e}")
    sys.exit(0)
EOF

# Enhanced DeepLog 결과 (Top-K)
echo "📊 Enhanced DeepLog 결과 (Top-K):"
python3 << 'EOF'
import pandas as pd
import json
import sys

try:
    with open('demo_enhanced_deeplog_output/enhanced_topk/deeplog_enhanced_summary.json', 'r') as f:
        summary = json.load(f)

    print(f"  총 시퀀스: {summary['total_sequences']:,}개")
    print(f"  실패 시퀀스: {summary['total_failures']:,}개")
    print(f"  노벨티 발견: {summary['total_novels']:,}개")
    print(f"  실제 알림: {summary['total_alerts']:,}개")

    if summary.get('alert_breakdown'):
        print(f"  알림 유형:")
        for alert_type, count in summary['alert_breakdown'].items():
            print(f"    - {alert_type}: {count}개")

    # 감소율 계산
    df_basic = pd.read_parquet('demo_enhanced_deeplog_output/processed/deeplog_infer.parquet')
    basic_violations = len(df_basic[df_basic['in_topk'] == False])
    enhanced_alerts = summary['total_alerts']

    reduction = (basic_violations - enhanced_alerts) / basic_violations * 100 if basic_violations > 0 else 0
    print(f"\n  ✨ 알림 감소율: {reduction:.1f}% (기본 대비)")
    print()
except Exception as e:
    print(f"  ❌ 파일 읽기 실패: {e}")
    sys.exit(0)
EOF

# Enhanced DeepLog 결과 (Top-P)
echo "📊 Enhanced DeepLog 결과 (Top-P):"
python3 << 'EOF'
import pandas as pd
import json
import sys

try:
    with open('demo_enhanced_deeplog_output/enhanced_topp/deeplog_enhanced_summary.json', 'r') as f:
        summary = json.load(f)

    print(f"  총 시퀀스: {summary['total_sequences']:,}개")
    print(f"  실패 시퀀스: {summary['total_failures']:,}개")
    print(f"  노벨티 발견: {summary['total_novels']:,}개")
    print(f"  실제 알림: {summary['total_alerts']:,}개")

    if summary.get('alert_breakdown'):
        print(f"  알림 유형:")
        for alert_type, count in summary['alert_breakdown'].items():
            print(f"    - {alert_type}: {count}개")
    print()
except Exception as e:
    print(f"  ❌ 파일 읽기 실패: {e}")
    sys.exit(0)
EOF

echo "======================================"
echo "🎉 Enhanced DeepLog 데모 완료!"
echo "======================================"
echo ""
echo "📁 생성된 파일들:"
echo "  📝 합성 로그: $DEMO_DIR/demo.log"
echo "  📝 전처리 결과: $DEMO_DIR/processed/parsed.parquet"
echo "  📝 DeepLog 모델: $DEMO_DIR/deeplog_model.pth"
echo "  📝 기본 추론: $DEMO_DIR/processed/deeplog_infer.parquet"
echo "  📝 Enhanced (Top-K):"
echo "     - 상세 결과: $DEMO_DIR/enhanced_topk/deeplog_enhanced_detailed.parquet"
echo "     - 알림 목록: $DEMO_DIR/enhanced_topk/deeplog_enhanced_alerts.parquet"
echo "     - 요약 정보: $DEMO_DIR/enhanced_topk/deeplog_enhanced_summary.json"
echo "  📝 Enhanced (Top-P):"
echo "     - 상세 결과: $DEMO_DIR/enhanced_topp/deeplog_enhanced_detailed.parquet"
echo "     - 알림 목록: $DEMO_DIR/enhanced_topp/deeplog_enhanced_alerts.parquet"
echo "     - 요약 정보: $DEMO_DIR/enhanced_topp/deeplog_enhanced_summary.json"
echo ""

echo "🔍 알림 상세 보기:"
echo "  cat $DEMO_DIR/enhanced_topk/deeplog_enhanced_summary.json | python3 -m json.tool"
echo ""

echo "💡 Enhanced DeepLog 특징:"
echo "  ✨ Top-K/Top-P 선택 가능"
echo "  ✨ K-of-N 슬라이딩 윈도우 판정"
echo "  ✨ 쿨다운으로 알림 폭주 방지"
echo "  ✨ 노벨티 탐지 및 집계"
echo "  ✨ 엔티티별 세션 관리"
echo "  ✨ 알림 시그니처 기반 중복 억제"
echo ""

echo "🎯 다음 단계:"
echo "  - 실제 로그로 테스트: study-preprocess deeplog-infer-enhanced --help"
echo "  - 파라미터 튜닝: k-of-n-k, cooldown-seq 조정"
echo "  - Top-P 실험: --top-p 0.8, 0.9, 0.95 비교"
echo ""
