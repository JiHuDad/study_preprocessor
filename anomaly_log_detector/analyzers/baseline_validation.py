#!/usr/bin/env python3  # 실행 스크립트 셔뱅  
"""Baseline 품질 검증 도구 요약

- 목적: baseline 후보(보통 parsed.parquet) 파일들의 품질을 사전 점검/필터링
- 방법: 템플릿 분포, 에러/경고 비율, 시간 일관성 등 다수 휴리스틱으로 점수화
- 출력: 각 파일의 유효성/점수/문제 목록/통계 및 리포트 생성
"""  # 모듈 요약 설명
import pandas as pd  # 데이터프레임 처리
import numpy as np  # 수치 연산
from pathlib import Path  # 경로 처리
from typing import List, Dict, Tuple, Optional  # 타입 힌트
import argparse  # CLI 인자 파서
import json  # JSON 입출력

class BaselineValidator:  # baseline 품질 검증기
    def __init__(self):  # 초기화
        # 정상 baseline의 기준값들 (경험적 임계값)  # 튜닝 가능
        self.thresholds = {
            'max_error_rate': 0.02,      # 2% 이하 에러율
            'max_warning_rate': 0.05,    # 5% 이하 경고율  
            'min_template_count': 10,    # 최소 10개 템플릿
            'max_rare_template_ratio': 0.3,  # 희귀 템플릿 30% 이하
            'min_log_count': 100,        # 최소 100개 로그
            'max_template_entropy': 10,  # 템플릿 엔트로피 상한
            'min_template_entropy': 2,   # 템플릿 엔트로피 하한
        }
    
    def validate_single_baseline(self, parsed_file: str) -> Dict:  # 단일 파일 품질 검증
        """단일 baseline 파일을 검증합니다."""  # API 설명
        try:
            df = pd.read_parquet(parsed_file)  # Parquet 로드
            file_name = Path(parsed_file).name  # 파일명
            
            # 기본 통계 계산  # 데이터 존재성 확인
            total_logs = len(df)  # 총 로그 수
            if total_logs == 0:  # 비어있으면 바로 실패
                return {
                    'file': file_name,
                    'valid': False,
                    'score': 0.0,
                    'issues': ['Empty dataset'],
                    'stats': {}
                }
            
            # 템플릿 분석  # 분포/희귀도 측정
            template_counts = df['template_id'].value_counts()  # 템플릿별 빈도
            unique_templates = len(template_counts)  # 고유 템플릿 수
            rare_templates = len([t for t, count in template_counts.items() if count == 1])  # 1회 등장 템플릿 수
            rare_template_ratio = rare_templates / max(unique_templates, 1)  # 희귀 템플릿 비율
            
            # 템플릿 분포 엔트로피  # 다양성 척도
            template_probs = template_counts / total_logs  # 확률 분포
            template_entropy = -sum(p * np.log2(p) for p in template_probs if p > 0)  # 엔트로피 계산
            
            # 에러/경고 로그 분석  # 키워드 기반 비율 추정
            error_keywords = ['error', 'ERROR', 'fail', 'FAIL', 'exception', 'Exception', 'panic', 'fatal', 'crash']  # 에러 키워드
            warning_keywords = ['warn', 'WARN', 'warning', 'WARNING', 'deprecated']  # 경고 키워드
            
            error_logs = df[df['raw'].str.contains('|'.join(error_keywords), case=False, na=False)]  # 에러 포함 행
            warning_logs = df[df['raw'].str.contains('|'.join(warning_keywords), case=False, na=False)]  # 경고 포함 행
            
            error_rate = len(error_logs) / total_logs  # 에러율
            warning_rate = len(warning_logs) / total_logs  # 경고율
            
            # 시간 일관성 검사  # 큰 시간 갭 탐지
            time_consistency = True  # 기본 True
            time_gaps = []  # 큰 갭 목록
            if 'timestamp' in df.columns:  # 타임스탬프 존재 시
                df_sorted = df.sort_values('timestamp').copy()  # 시간 정렬
                df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'], errors='coerce')  # datetime 변환
                valid_timestamps = df_sorted.dropna(subset=['timestamp'])  # 유효한 행만
                
                if len(valid_timestamps) > 1:  # 2개 이상일 때만
                    time_diffs = valid_timestamps['timestamp'].diff().dt.total_seconds()  # 연속 차이(초)
                    # 1시간 이상 갭이 있으면 의심  # 임계 3600초
                    large_gaps = time_diffs[time_diffs > 3600]  # 큰 갭만 추출
                    time_gaps = large_gaps.tolist()  # 리스트화
                    time_consistency = len(large_gaps) < total_logs * 0.1  # 10% 이하이면 일관성 양호
            
            # 로그 밀도 분석 (시간당 로그 수)  # 밀도 지표
            logs_per_hour = None  # 기본 None
            if 'timestamp' in df.columns and len(df) > 1:  # 계산 가능 조건
                try:
                    start_time = pd.to_datetime(df['timestamp'].min())  # 시작 시간
                    end_time = pd.to_datetime(df['timestamp'].max())  # 종료 시간
                    duration_hours = (end_time - start_time).total_seconds() / 3600  # 시간 차
                    if duration_hours > 0:
                        logs_per_hour = total_logs / duration_hours  # 시간당 로그 수
                except:
                    pass  # 변환 실패 시 무시
            
            # 통계 정리  # 리포트용 통계 묶음
            stats = {
                'total_logs': total_logs,
                'unique_templates': unique_templates,
                'rare_templates': rare_templates,
                'rare_template_ratio': rare_template_ratio,
                'template_entropy': template_entropy,
                'error_rate': error_rate,
                'warning_rate': warning_rate,
                'time_consistency': time_consistency,
                'large_time_gaps': len(time_gaps),
                'logs_per_hour': logs_per_hour
            }
            
            # 검증 결과 계산  # 감점 방식으로 점수 산출
            issues = []  # 이슈 목록
            score = 1.0  # 완벽한 점수에서 시작하여 문제마다 감점
            
            # 각 기준별 검사  # 임계 초과/미달 시 감점 및 이슈 기록
            if error_rate > self.thresholds['max_error_rate']:
                issues.append(f"높은 에러율: {error_rate:.2%} (기준: {self.thresholds['max_error_rate']:.2%})")
                score -= 0.3
            
            if warning_rate > self.thresholds['max_warning_rate']:
                issues.append(f"높은 경고율: {warning_rate:.2%} (기준: {self.thresholds['max_warning_rate']:.2%})")
                score -= 0.2
            
            if unique_templates < self.thresholds['min_template_count']:
                issues.append(f"템플릿 부족: {unique_templates}개 (기준: {self.thresholds['min_template_count']}개)")
                score -= 0.2
            
            if rare_template_ratio > self.thresholds['max_rare_template_ratio']:
                issues.append(f"희귀 템플릿 과다: {rare_template_ratio:.2%} (기준: {self.thresholds['max_rare_template_ratio']:.2%})")
                score -= 0.1
            
            if total_logs < self.thresholds['min_log_count']:
                issues.append(f"로그 수 부족: {total_logs}개 (기준: {self.thresholds['min_log_count']}개)")
                score -= 0.3
            
            if template_entropy > self.thresholds['max_template_entropy']:
                issues.append(f"템플릿 엔트로피 과다: {template_entropy:.2f} (기준: {self.thresholds['max_template_entropy']})")
                score -= 0.2
            
            if template_entropy < self.thresholds['min_template_entropy']:
                issues.append(f"템플릿 다양성 부족: {template_entropy:.2f} (기준: {self.thresholds['min_template_entropy']})")
                score -= 0.1
            
            if not time_consistency:
                issues.append("시간 일관성 문제: 큰 시간 갭이 다수 존재")
                score -= 0.1
            
            # 최종 점수 조정  # 하한 0.0 적용
            score = max(0.0, score)
            is_valid = score >= 0.7 and len(issues) <= 2  # 70% 이상 점수, 문제 2개 이하
            
            return {
                'file': file_name,
                'valid': is_valid,
                'score': score,
                'issues': issues,
                'stats': stats
            }
            
        except Exception as e:  # 처리 실패 시 결과
            return {
                'file': Path(parsed_file).name,
                'valid': False,
                'score': 0.0,
                'issues': [f'처리 오류: {str(e)}'],
                'stats': {}
            }
    
    def validate_multiple_baselines(self, baseline_files: List[str]) -> Dict:  # 다중 파일 검증
        """여러 baseline 파일들을 검증하고 필터링합니다."""  # API 설명
        
        print(f"🔍 {len(baseline_files)}개 baseline 파일 검증 중...")  # 진행 로그
        
        results = []  # 개별 결과 저장소
        for baseline_file in baseline_files:  # 파일별 반복
            print(f"   검증: {Path(baseline_file).name}")  # 파일 표시
            result = self.validate_single_baseline(baseline_file)  # 단일 검증
            results.append(result)  # 누적
        
        # 결과 분석  # 유효/무효 분리
        valid_baselines = [r for r in results if r['valid']]
        invalid_baselines = [r for r in results if not r['valid']]
        
        # 상호 일관성 검사 (valid한 것들끼리)  # 정상 후보 간 분산 체크
        if len(valid_baselines) > 1:
            consistency_issues = self._check_mutual_consistency(valid_baselines, baseline_files)
        else:
            consistency_issues = []  # 비교 불가 시 빈 리스트
        
        print(f"\n📊 검증 결과:")  # 요약 로그
        print(f"   ✅ 유효: {len(valid_baselines)}개")
        print(f"   ❌ 무효: {len(invalid_baselines)}개")
        print(f"   ⚠️  일관성 문제: {len(consistency_issues)}개")
        
        return {
            'total_files': len(baseline_files),  # 총 파일 수
            'valid_count': len(valid_baselines),  # 유효 개수
            'invalid_count': len(invalid_baselines),  # 무효 개수
            'valid_baselines': valid_baselines,  # 유효 리스트
            'invalid_baselines': invalid_baselines,  # 무효 리스트
            'consistency_issues': consistency_issues,  # 일관성 이슈
            'recommended_baselines': [r['file'] for r in valid_baselines if r['score'] >= 0.8]  # 추천 파일
        }
    
    def _check_mutual_consistency(self, valid_results: List[Dict], baseline_files: List[str]) -> List[Dict]:  # 상호 일관성 검사
        """유효한 baseline들 간의 상호 일관성을 검사합니다."""  # API 설명
        
        consistency_issues = []  # 이슈 누적
        
        # 주요 지표들의 분포 분석  # 비교 대상 메트릭
        metrics = ['error_rate', 'warning_rate', 'template_entropy', 'logs_per_hour']
        
        for metric in metrics:  # 메트릭별 반복
            values = []  # 값 수집
            for result in valid_results:  # 각 결과에서
                val = result['stats'].get(metric)  # 값 추출
                if val is not None:
                    values.append(val)  # 수집
            
            if len(values) < 2:  # 비교 불가 시 스킵
                continue
                
            mean_val = np.mean(values)  # 평균
            std_val = np.std(values)  # 표준편차
            
            # Outlier 탐지 (2σ 밖의 값들)  # 간단한 이상치 규칙
            for i, (result, val) in enumerate(zip(valid_results, values)):
                if abs(val - mean_val) > 2 * std_val:
                    consistency_issues.append({
                        'file': result['file'],  # 파일명
                        'metric': metric,  # 메트릭명
                        'value': val,  # 실제 값
                        'mean': mean_val,  # 평균
                        'std': std_val,  # 표준편차
                        'description': f'{metric}이 다른 baseline들과 {abs(val - mean_val)/max(mean_val, 1e-6)*100:.1f}% 차이'  # 설명
                    })
        
        return consistency_issues  # 이슈 목록 반환
    
    def generate_validation_report(self, validation_result: Dict, output_file: str = None) -> str:  # 리포트 생성
        """검증 결과 리포트를 생성합니다."""  # API 설명
        
        report = f"""# Baseline 파일 검증 리포트

## 📊 전체 요약
- **총 파일 수**: {validation_result['total_files']}개
- **유효한 파일**: {validation_result['valid_count']}개
- **무효한 파일**: {validation_result['invalid_count']}개
- **추천 파일**: {len(validation_result['recommended_baselines'])}개

## ✅ 유효한 Baseline 파일들

"""
        
        for result in validation_result['valid_baselines']:  # 유효 파일 섹션
            stats = result['stats']  # 통계
            report += f"""### {result['file']}
- **품질 점수**: {result['score']:.2f}/1.0
- **로그 수**: {stats.get('total_logs', 0):,}개
- **템플릿 수**: {stats.get('unique_templates', 0)}개
- **에러율**: {stats.get('error_rate', 0):.2%}
- **경고율**: {stats.get('warning_rate', 0):.2%}
- **템플릿 엔트로피**: {stats.get('template_entropy', 0):.2f}
"""
            if result['issues']:  # 경미 이슈 나열
                report += f"- **경미한 이슈**: {', '.join(result['issues'])}\n"
            report += "\n"  # 공백 라인
        
        if validation_result['invalid_baselines']:  # 무효 파일 섹션
            report += "## ❌ 무효한 Baseline 파일들\n\n"
            for result in validation_result['invalid_baselines']:
                report += f"""### {result['file']}
- **품질 점수**: {result['score']:.2f}/1.0
- **주요 문제**: {', '.join(result['issues'])}

"""
        
        if validation_result['consistency_issues']:  # 일관성 이슈 섹션
            report += "## ⚠️  일관성 문제\n\n"
            for issue in validation_result['consistency_issues']:
                report += f"- **{issue['file']}**: {issue['description']}\n"
            report += "\n"
        
        report += f"""## 💡 권장사항

### 사용 권장 파일들
{chr(10).join('- ' + f for f in validation_result['recommended_baselines'])}

### 품질 개선 방법
1. **에러율이 높은 파일들**: 해당 시기의 시스템 상태 확인
2. **로그 수가 적은 파일들**: 더 긴 기간의 로그 수집
3. **템플릿 다양성 문제**: 다양한 시스템 활동이 포함된 기간 선택
4. **시간 일관성 문제**: 로그 수집 중단 구간 확인

### 이상탐지 정확도 향상을 위한 조치
- 품질 점수 0.8 이상 파일들만 baseline으로 사용
- 최소 3개 이상의 유효한 baseline 확보
- 일관성 문제가 있는 파일들은 제외 고려
"""
        
        if output_file:  # 파일로 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 검증 리포트 저장: {output_file}")
        
        return report  # 리포트 문자열 반환

def validate_baseline_files(baseline_files: List[str], output_dir: str = None):  # 상위 헬퍼
    """Baseline 파일들을 검증하고 리포트를 생성합니다."""  # API 설명
    
    if output_dir:  # 출력 디렉토리 준비
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    validator = BaselineValidator()  # 검증기 생성
    
    # 검증 실행  # 다중 파일 처리
    result = validator.validate_multiple_baselines(baseline_files)
    
    # 결과 저장  # 옵션에 따라 파일 저장
    if output_dir:
        # JSON 결과 저장
        with open(output_path / "validation_result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        # 리포트 생성
        report = validator.generate_validation_report(result, output_path / "baseline_validation_report.md")
        
        print(f"📂 결과 저장 위치: {output_path}")
    else:
        validator.generate_validation_report(result)  # 콘솔 출력용
    
    return result  # 결과 딕셔너리 반환

def main():  # CLI 엔트리 포인트
    parser = argparse.ArgumentParser(description="Baseline 파일 품질 검증 도구")  # 인자 파서
    parser.add_argument("baseline_files", nargs='+', help="검증할 baseline 파일들 (parsed.parquet)")  # 입력 파일들
    parser.add_argument("--output-dir", help="결과 출력 디렉토리")  # 출력 폴더
    parser.add_argument("--score-threshold", type=float, default=0.7, help="유효 baseline 최소 점수 (기본: 0.7)")  # (미사용) 예비 옵션
    
    args = parser.parse_args()  # 인자 파싱
    
    # 임계값 업데이트  # 필요 시 thresholds를 조정 가능 (현재는 기본값 사용)
    validator = BaselineValidator()  # 인스턴스 생성
    
    result = validate_baseline_files(args.baseline_files, args.output_dir)  # 검증 실행
    
    print(f"\n🎯 권장 baseline 파일들:")  # 권장 목록 출력
    for file in result['recommended_baselines']:
        print(f"   ✅ {file}")

if __name__ == "__main__":  # 직접 실행 시
    main()  # 메인 호출
