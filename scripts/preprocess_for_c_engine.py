#!/usr/bin/env python3
"""
로그를 C inference engine용으로 전처리

C 엔진의 간단한 템플릿 매칭을 돕기 위해, Python의 정확한 마스킹을 사용합니다.
"""

import sys
import argparse
from pathlib import Path

# anomaly_log_detector 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from anomaly_log_detector.preprocess import mask_message


def preprocess_log_for_c_engine(input_file: str, output_file: str) -> None:
    """
    로그를 C 엔진용으로 전처리 (마스킹만 수행)

    Args:
        input_file: 입력 로그 파일 경로
        output_file: 출력 파일 경로 (마스킹된 로그)
    """
    processed_count = 0

    print(f"📂 입력: {input_file}")
    print(f"📝 출력: {output_file}")
    print("🔄 마스킹 중...")

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_no, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Python의 mask_message 사용 (정확한 마스킹)
                masked = mask_message(line)
                f_out.write(masked + '\n')
                processed_count += 1

                if processed_count % 1000 == 0:
                    print(f"  처리됨: {processed_count} 라인")

            except Exception as e:
                print(f"⚠️  라인 {line_no} 처리 실패: {e}")
                # 실패 시 원본 출력
                f_out.write(line + '\n')

    print(f"✅ 완료! 총 {processed_count} 라인 처리")


def main():
    parser = argparse.ArgumentParser(
        description='로그를 C inference engine용으로 전처리 (마스킹)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 사용
  python preprocess_for_c_engine.py -i /var/log/syslog -o /tmp/masked.log

  # C 엔진으로 추론
  ./bin/inference_engine -d models/deeplog.onnx -v models/vocab.json -i /tmp/masked.log

  # 파이프라인
  python preprocess_for_c_engine.py -i input.log -o masked.log && \\
  ./bin/inference_engine -d models/deeplog.onnx -v models/vocab.json -i masked.log
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='입력 로그 파일 경로'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='출력 파일 경로 (마스킹된 로그)'
    )

    args = parser.parse_args()

    # 입력 파일 확인
    if not Path(args.input).exists():
        print(f"❌ 오류: 입력 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    # 출력 디렉토리 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        preprocess_log_for_c_engine(args.input, args.output)
    except Exception as e:
        print(f"❌ 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
