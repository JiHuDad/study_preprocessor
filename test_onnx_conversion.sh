#!/bin/bash
# ONNX 변환 테스트 스크립트

set -e

echo "======================================"
echo "ONNX 변환 테스트"
echo "======================================"
echo ""

# 테스트 디렉토리 준비
TEST_DIR="test_onnx_output"
echo "🧹 테스트 디렉토리 준비..."
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# 1. 간단한 합성 로그 생성
echo "1️⃣  합성 로그 생성..."
study-preprocess gen-synth \
  --out "$TEST_DIR/test.log" \
  --lines 500 \
  --anomaly-rate 0.05

# 2. 전처리
echo "2️⃣  전처리..."
study-preprocess parse \
  --input "$TEST_DIR/test.log" \
  --out-dir "$TEST_DIR/processed"

# 3. DeepLog 입력 생성
echo "3️⃣  DeepLog 입력 생성..."
study-preprocess build-deeplog \
  --parsed "$TEST_DIR/processed/parsed.parquet" \
  --out-dir "$TEST_DIR/processed"

# 4. DeepLog 학습 (빠른 테스트)
echo "4️⃣  DeepLog 학습..."
study-preprocess deeplog-train \
  --seq "$TEST_DIR/processed/sequences.parquet" \
  --vocab "$TEST_DIR/processed/vocab.json" \
  --out "$TEST_DIR/deeplog.pth" \
  --seq-len 20 \
  --epochs 2

# 5. ONNX 변환 (핵심 테스트!)
echo "5️⃣  ONNX 변환..."
study-preprocess convert-onnx \
  --deeplog-model "$TEST_DIR/deeplog.pth" \
  --vocab "$TEST_DIR/processed/vocab.json" \
  --output-dir "$TEST_DIR/onnx" \
  --validate

echo ""
echo "======================================"
echo "✅ ONNX 변환 테스트 완료!"
echo "======================================"
echo ""

# 결과 확인
if [ -f "$TEST_DIR/onnx/deeplog.onnx" ]; then
    echo "📊 생성된 파일:"
    ls -lh "$TEST_DIR/onnx/"
    echo ""

    echo "📄 메타데이터:"
    if [ -f "$TEST_DIR/onnx/deeplog.onnx.meta.json" ]; then
        cat "$TEST_DIR/onnx/deeplog.onnx.meta.json" | python3 -m json.tool
    fi
    echo ""

    echo "✨ ONNX 모델 성공적으로 생성되었습니다!"
    echo "   모델 경로: $TEST_DIR/onnx/deeplog.onnx"
    echo ""
    echo "💡 다음 단계:"
    echo "   - C 바이너리로 추론: cd hybrid_system/inference && make"
    echo "   - Python에서 ONNX 검증: python -c \"import onnx; onnx.checker.check_model('$TEST_DIR/onnx/deeplog.onnx')\""
else
    echo "❌ ONNX 파일이 생성되지 않았습니다."
    exit 1
fi
