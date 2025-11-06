#!/bin/bash

echo "=== DeepLog 문제 진단 ==="
echo ""

# 1. 파일 존재 확인
echo "1️⃣  파일 존재 확인"
echo "Models:"
ls -lh models_*/vocab.json models_*/deeplog.pth models_*/drain3_state.json 2>/dev/null || echo "  ❌ 모델 파일 없음"
echo ""
echo "Inference:"
ls -lh inference_*/deeplog_infer.parquet inference_*/sequences.parquet 2>/dev/null || echo "  ❌ 추론 파일 없음"
echo ""

# 2. vocab 크기 확인
echo "2️⃣  Vocab 크기 비교"
python3 << 'PYEOF'
import json
from pathlib import Path
import glob

model_vocabs = glob.glob("models_*/vocab.json")
infer_vocabs = glob.glob("inference_*/vocab.json")

if model_vocabs:
    with open(model_vocabs[0]) as f:
        model_vocab = json.load(f)
    print(f"  학습 vocab 크기: {len(model_vocab)}")
else:
    print("  ❌ 학습 vocab 없음")

if infer_vocabs:
    with open(infer_vocabs[0]) as f:
        infer_vocab = json.load(f)
    print(f"  추론 vocab 크기: {len(infer_vocab)}")

    if model_vocabs:
        if model_vocab == infer_vocab:
            print("  ✅ Vocab 동일")
        else:
            print("  ❌ Vocab 불일치!")
else:
    print("  ℹ️  추론 vocab 없음 (모델 vocab 사용)")
PYEOF
echo ""

# 3. 실제 예측 결과 확인
echo "3️⃣  DeepLog 예측 결과"
python3 << 'PYEOF'
import pandas as pd
import glob

infer_files = glob.glob("inference_*/deeplog_infer.parquet")
if not infer_files:
    print("  ❌ deeplog_infer.parquet 없음")
    exit(0)

df = pd.read_parquet(infer_files[0])
print(f"  전체 시퀀스: {len(df)}")

if 'prediction_ok' in df.columns:
    col = 'prediction_ok'
    failures = (df[col] == False).sum()
elif 'in_topk' in df.columns:
    col = 'in_topk'
    failures = (df[col] == False).sum()
else:
    print("  ❌ 예측 결과 컬럼 없음")
    exit(0)

print(f"  예측 실패: {failures} ({failures/len(df)*100:.1f}%)")
print(f"  예측 성공: {len(df)-failures} ({(len(df)-failures)/len(df)*100:.1f}%)")

# 템플릿 문자열 확인
if 'target_template' in df.columns:
    print("  ✅ 템플릿 문자열 있음")
    print(f"\n  실패 샘플 예시:")
    violations = df[df[col] == False].head(3)
    for idx, row in violations.iterrows():
        print(f"    - 실제: {row.get('target_template', 'N/A')[:50]}")
        print(f"      예측: {row.get('predicted_templates', 'N/A')[:80]}")
else:
    print("  ❌ 템플릿 문자열 없음")
PYEOF
echo ""

# 4. 알림 결과 확인
echo "4️⃣  DeepLog 알림 (K-of-N 필터링 후)"
python3 << 'PYEOF'
import pandas as pd
import glob

alert_files = glob.glob("inference_*/deeplog_alerts.parquet")
if not alert_files:
    print("  ❌ deeplog_alerts.parquet 없음")
    exit(0)

alerts = pd.read_parquet(alert_files[0])
print(f"  발생 알림: {len(alerts)}개")
if len(alerts) > 0:
    print(f"  알림 유형:")
    for atype, count in alerts['alert_type'].value_counts().items():
        print(f"    - {atype}: {count}개")
PYEOF
echo ""

echo "=== 진단 완료 ==="
