#!/bin/bash

# 배치 로그 분석 실행 스크립트 (Wrapper)
#
# ⚠️  이 스크립트는 run_enhanced_batch_analysis.sh로 리디렉션하는 wrapper입니다.
# ⚠️  새로운 기능은 run_enhanced_batch_analysis.sh를 사용하세요.
#
# 사용법: ./run_batch_analysis.sh <로그디렉토리> [target파일] [작업디렉토리]

echo "⚠️  주의: run_batch_analysis.sh는 이제 run_enhanced_batch_analysis.sh의 wrapper입니다."
echo "   향후 버전에서는 이 wrapper가 제거될 수 있습니다."
echo "   직접 run_enhanced_batch_analysis.sh를 사용하는 것을 권장합니다."
echo ""

# run_enhanced_batch_analysis.sh로 모든 인자 전달
exec ./run_enhanced_batch_analysis.sh "$@"
