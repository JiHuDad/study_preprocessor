#!/usr/bin/env python3
"""
MS-CRED Analyzer (Wrapper)

⚠️  이 스크립트는 이제 anomaly_log_detector.analyzers.mscred_analysis의 wrapper입니다.
⚠️  향후 버전에서는 제거될 수 있습니다.
⚠️  대신 CLI 명령어 사용을 권장합니다: alog-detect analyze-mscred

호환성을 위해 제공되는 wrapper입니다.
"""
import sys
import warnings

# 경고 메시지 출력
warnings.warn(
    "mscred_analyzer.py는 deprecated되었습니다. "
    "'alog-detect analyze-mscred' 명령어를 사용하세요.",
    DeprecationWarning,
    stacklevel=2
)

print("⚠️  주의: mscred_analyzer.py는 이제 deprecated되었습니다.")
print("   대신 'alog-detect analyze-mscred' 명령어를 사용하세요.")
print()

# 실제 모듈로 리디렉션
from anomaly_log_detector.analyzers.mscred_analysis import main

if __name__ == "__main__":
    sys.exit(main())
