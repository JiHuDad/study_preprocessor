#!/bin/bash

# 전처리 테스트 스크립트
# 사용법: ./test_preprocessing.sh

set -e

echo "🧪 전처리 테스트 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 가상환경 활성화
if [ -f ".venv/bin/activate" ]; then
    echo "🔵 .venv 가상환경 활성화 중..."
    source .venv/bin/activate
    echo "✅ 가상환경 활성화됨: $VIRTUAL_ENV"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "🔵 기존 가상환경 감지됨: $VIRTUAL_ENV"
else
    echo "⚠️  가상환경이 없습니다. python3를 사용합니다."
fi

# Python 명령어 설정
PYTHON_CMD="python"
if [ -z "$VIRTUAL_ENV" ]; then
    PYTHON_CMD="python3"
fi

echo ""
echo "🔍 환경 확인"
echo "   - Python: $PYTHON_CMD"
echo "   - 버전: $($PYTHON_CMD --version)"
echo "   - 위치: $(which $PYTHON_CMD)"

# 패키지 확인
echo ""
echo "🔍 패키지 확인"
packages=("pandas" "torch" "drain3" "study_preprocessor")
for pkg in "${packages[@]}"; do
    if $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
        version=$($PYTHON_CMD -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        echo "   ✅ $pkg ($version)"
    else
        echo "   ❌ $pkg (누락)"
    fi
done

# 테스트 로그 생성
echo ""
echo "🔧 테스트 로그 생성"
TEST_LOG="test_sample.log"
cat > "$TEST_LOG" << 'EOF'
2024-01-01 10:00:01 kernel: [    0.000000] Linux version 5.15.0-generic
2024-01-01 10:00:02 kernel: [    0.000001] Command line: BOOT_IMAGE=/boot/vmlinuz
2024-01-01 10:00:03 systemd[1]: Starting Network Manager...
2024-01-01 10:00:04 NetworkManager[1234]: <info> NetworkManager (version 1.30.0) is starting...
2024-01-01 10:00:05 kernel: [    1.234567] USB disconnect, address 1
2024-01-01 10:00:06 systemd[1]: Started Network Manager.
2024-01-01 10:00:07 kernel: [    2.345678] USB connect, address 2
2024-01-01 10:00:08 NetworkManager[1234]: <info> device eth0: carrier is on
2024-01-01 10:00:09 systemd[1]: Reached target Network.
2024-01-01 10:00:10 kernel: [    3.456789] Memory: 8192MB available
EOF

echo "✅ 테스트 로그 생성됨: $TEST_LOG ($(wc -l < "$TEST_LOG") 라인)"

# 테스트 디렉토리 생성
TEST_DIR="test_preprocessing_output"
mkdir -p "$TEST_DIR"

echo ""
echo "🚀 전처리 실행"
echo "   입력: $TEST_LOG"
echo "   출력: $TEST_DIR"
echo "   명령어: $PYTHON_CMD -m study_preprocessor.cli parse --input \"$TEST_LOG\" --out-dir \"$TEST_DIR\""

# 전처리 실행
if $PYTHON_CMD -m study_preprocessor.cli parse \
    --input "$TEST_LOG" \
    --out-dir "$TEST_DIR" 2>&1; then
    echo "✅ 전처리 명령어 실행 성공"
else
    echo "❌ 전처리 명령어 실행 실패"
    exit 1
fi

echo ""
echo "🔍 결과 확인"
if [ -f "$TEST_DIR/parsed.parquet" ]; then
    echo "✅ parsed.parquet 파일 생성됨"
    
    # 파일 정보 출력
    size=$(stat -c%s "$TEST_DIR/parsed.parquet" 2>/dev/null || echo "0")
    echo "   - 크기: $(echo $size | numfmt --to=iec)"
    
    # 레코드 수 확인
    records=$($PYTHON_CMD -c "import pandas as pd; df=pd.read_parquet('$TEST_DIR/parsed.parquet'); print(len(df))" 2>/dev/null || echo "확인 불가")
    echo "   - 레코드 수: $records"
    
    # 컬럼 확인
    columns=$($PYTHON_CMD -c "import pandas as pd; df=pd.read_parquet('$TEST_DIR/parsed.parquet'); print(', '.join(df.columns))" 2>/dev/null || echo "확인 불가")
    echo "   - 컬럼: $columns"
    
    echo "✅ 전처리 테스트 성공!"
else
    echo "❌ parsed.parquet 파일이 생성되지 않았습니다."
    echo ""
    echo "🔍 생성된 파일들:"
    ls -la "$TEST_DIR/" 2>/dev/null || echo "   디렉토리가 비어있습니다."
    exit 1
fi

echo ""
echo "🧹 정리"
rm -f "$TEST_LOG"
rm -rf "$TEST_DIR"
echo "✅ 테스트 파일 정리 완료"

echo ""
echo "🎉 전처리 테스트 완료!"
echo "   이제 train_models.sh를 안전하게 실행할 수 있습니다."
