#!/usr/bin/env python3
"""
anomaly_log_detector 패키지의 메인 엔트리포인트
python -m anomaly_log_detector.cli 명령어를 지원하기 위한 파일
"""

from .cli import main

if __name__ == "__main__":
    main()
