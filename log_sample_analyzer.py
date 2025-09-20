#!/usr/bin/env python3
"""
이상 로그 샘플 추출 및 분석 도구
- 이상탐지 결과에서 실제 문제가 되는 로그들을 추출
- 사람이 이해하기 쉬운 형태로 로그 샘플과 분석 제공
- 전후 맥락과 함께 이상 패턴 설명
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime

class LogSampleAnalyzer:
    def __init__(self):
        self.max_samples_per_type = 5  # 타입별 최대 샘플 수
        self.context_lines = 3  # 전후 맥락 라인 수
        
    def extract_baseline_anomaly_samples(self, parsed_file: str, baseline_scores_file: str) -> Dict:
        """베이스라인 이상탐지에서 문제가 되는 로그 샘플들을 추출합니다."""
        
        # 데이터 로드
        df_parsed = pd.read_parquet(parsed_file)
        df_scores = pd.read_parquet(baseline_scores_file)
        
        # 이상 윈도우만 필터링
        anomaly_windows = df_scores[df_scores['is_anomaly'] == True].copy()
        if len(anomaly_windows) == 0:
            return {'type': 'baseline', 'anomaly_count': 0, 'samples': []}
        
        # 점수 내림차순 정렬
        anomaly_windows = anomaly_windows.sort_values('score', ascending=False)
        
        samples = []
        for _, window_row in anomaly_windows.head(self.max_samples_per_type).iterrows():
            window_start = int(window_row['window_start_line'])
            window_size = 50  # 기본 윈도우 크기
            
            # 윈도우 내의 로그들 추출
            window_logs = df_parsed[
                (df_parsed['line_no'] >= window_start) & 
                (df_parsed['line_no'] < window_start + window_size)
            ].copy()
            
            if len(window_logs) == 0:
                continue
            
            # 전후 맥락 로그 추출
            context_before = df_parsed[
                (df_parsed['line_no'] >= max(0, window_start - self.context_lines)) & 
                (df_parsed['line_no'] < window_start)
            ].copy()
            
            context_after = df_parsed[
                (df_parsed['line_no'] >= window_start + window_size) & 
                (df_parsed['line_no'] < window_start + window_size + self.context_lines)
            ].copy()
            
            # 윈도우 내 이상 패턴 분석
            analysis = self._analyze_window_patterns(window_logs, window_row)
            
            # 대표 로그들 선별
            representative_logs = self._select_representative_logs(window_logs, analysis)
            
            sample = {
                'window_id': f"window_{window_start}",
                'window_start_line': window_start,
                'anomaly_score': float(window_row['score']),
                'unseen_rate': float(window_row['unseen_rate']),
                'freq_z_score': float(window_row.get('freq_z', 0)),
                'time_range': {
                    'start': str(window_logs['timestamp'].min()) if 'timestamp' in window_logs.columns else None,
                    'end': str(window_logs['timestamp'].max()) if 'timestamp' in window_logs.columns else None,
                },
                'analysis': analysis,
                'representative_logs': representative_logs,
                'context_before': self._format_log_entries(context_before),
                'context_after': self._format_log_entries(context_after),
                'total_logs_in_window': len(window_logs)
            }
            
            samples.append(sample)
        
        return {
            'type': 'baseline',
            'method': 'Window-based frequency and novelty detection',
            'anomaly_count': len(anomaly_windows),
            'analyzed_count': len(samples),
            'samples': samples
        }
    
    def extract_deeplog_anomaly_samples(self, parsed_file: str, deeplog_infer_file: str, 
                                      vocab_file: str, seq_len: int = 50) -> Dict:
        """DeepLog 이상탐지에서 문제가 되는 로그 샘플들을 추출합니다."""
        
        # 데이터 로드
        df_parsed = pd.read_parquet(parsed_file)
        df_infer = pd.read_parquet(deeplog_infer_file)
        
            with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        # 역방향 매핑 생성
        idx_to_template = {v: k for k, v in vocab.items()}
        
        # 예측 실패한 경우들 필터링
        anomalies = df_infer[df_infer['in_topk'] == False].copy()
        if len(anomalies) == 0:
            return {'type': 'deeplog', 'anomaly_count': 0, 'samples': []}
        
        samples = []
        for _, anomaly_row in anomalies.head(self.max_samples_per_type).iterrows():
            seq_idx = int(anomaly_row['idx'])
            predicted_template = idx_to_template.get(int(anomaly_row['target']), 'UNKNOWN')
            
            # 시퀀스 위치를 라인 번호로 변환
            approx_line_no = seq_idx + seq_len
            
            # 해당 라인 주변의 로그들 찾기
            target_logs = df_parsed[
                (df_parsed['line_no'] >= approx_line_no - 5) & 
                (df_parsed['line_no'] <= approx_line_no + 5)
            ].copy()
            
            if len(target_logs) == 0:
                continue
            
            # 가장 가까운 로그 찾기
            target_log = target_logs.iloc[len(target_logs)//2] if len(target_logs) > 0 else None
            if target_log is None:
                continue
            
            # 시퀀스 맥락 (이전 seq_len개 로그들)
            sequence_context = df_parsed[
                (df_parsed['line_no'] >= max(0, approx_line_no - seq_len)) & 
                (df_parsed['line_no'] < approx_line_no)
            ].copy()
            
            # 시퀀스 분석
            sequence_analysis = self._analyze_sequence_patterns(sequence_context, target_log, predicted_template)
            
            sample = {
                'sequence_id': f"seq_{seq_idx}",
                'sequence_index': seq_idx,
                'predicted_template_id': predicted_template,
                'actual_line_no': int(target_log['line_no']),
                'actual_template_id': str(target_log['template_id']),
                'timestamp': str(target_log['timestamp']) if target_log['timestamp'] else None,
                'analysis': sequence_analysis,
                'target_log': {
                    'line_no': int(target_log['line_no']),
                    'raw_message': str(target_log['raw']),
                    'masked_message': str(target_log['masked']),
                    'template': str(target_log['template']),
                    'template_id': str(target_log['template_id'])
                },
                'sequence_context': self._format_log_entries(sequence_context),
                'sequence_length': len(sequence_context)
            }
            
            samples.append(sample)
        
        return {
            'type': 'deeplog',
            'method': 'LSTM sequence prediction (top-k violation)',
            'anomaly_count': len(anomalies),
            'analyzed_count': len(samples),
            'samples': samples
        }
    
    def extract_comparative_anomaly_samples(self, comparative_anomalies_file: str, 
                                          parsed_file: str) -> Dict:
        """비교 분석에서 발견된 이상 로그 샘플들을 추출합니다."""
        
        # 비교 분석 결과 로드
        with open(comparative_anomalies_file, 'r', encoding='utf-8') as f:
            anomalies = json.load(f)
        
        if not anomalies:
            return {'type': 'comparative', 'anomaly_count': 0, 'samples': []}
        
        # parsed 파일 로드
        df_parsed = pd.read_parquet(parsed_file)
        
        samples = []
        for anomaly in anomalies[:self.max_samples_per_type]:
            analysis_type = anomaly.get('type', 'unknown')
            
            if 'template' in analysis_type.lower():
                # 템플릿 관련 이상
                sample = self._extract_template_anomaly_sample(anomaly, df_parsed)
            elif 'metric' in analysis_type.lower():
                # 메트릭 관련 이상
                sample = self._extract_metric_anomaly_sample(anomaly, df_parsed)
            else:
                # 기타 이상
                sample = self._extract_general_anomaly_sample(anomaly, df_parsed)
            
            if sample:
                samples.append(sample)
        
        return {
            'type': 'comparative',
            'method': 'Cross-file pattern comparison',
            'anomaly_count': len(anomalies),
            'analyzed_count': len(samples),
            'samples': samples
        }
    
    def _analyze_window_patterns(self, window_logs: pd.DataFrame, window_row: pd.Series) -> Dict:
        """윈도우 내 이상 패턴을 분석합니다."""
        
        # 템플릿 분포 분석
        template_counts = window_logs['template_id'].value_counts()
        
        # 새로운 템플릿 식별
        unseen_rate = float(window_row['unseen_rate'])
        freq_z = float(window_row.get('freq_z', 0))
        
        # 에러/경고 키워드 분석
        error_keywords = ['error', 'ERROR', 'fail', 'FAIL', 'exception', 'Exception', 'panic', 'fatal', 'crash']
        warning_keywords = ['warn', 'WARN', 'warning', 'WARNING', 'deprecated']
        
        error_logs = window_logs[window_logs['raw'].str.contains('|'.join(error_keywords), case=False, na=False)]
        warning_logs = window_logs[window_logs['raw'].str.contains('|'.join(warning_keywords), case=False, na=False)]
        
        # 시간 밀도 분석
        time_density = self._analyze_time_density(window_logs)
        
        analysis = {
            'dominant_templates': [
                {'template_id': str(tid), 'count': int(count), 'percentage': count/len(window_logs)*100}
                for tid, count in template_counts.head(3).items()
            ],
            'unseen_templates_ratio': unseen_rate,
            'frequency_spike_score': freq_z,
            'error_log_count': len(error_logs),
            'warning_log_count': len(warning_logs),
            'error_percentage': len(error_logs) / len(window_logs) * 100,
            'time_density': time_density,
            'anomaly_indicators': self._identify_anomaly_indicators(window_logs, unseen_rate, freq_z, error_logs, warning_logs)
        }
        
        return analysis
    
    def _analyze_sequence_patterns(self, sequence_context: pd.DataFrame, target_log: pd.Series, 
                                 predicted_template: str) -> Dict:
        """시퀀스 패턴을 분석합니다."""
        
        # 시퀀스 내 템플릿 전환 패턴
        template_sequence = sequence_context['template_id'].tolist() + [target_log['template_id']]
        template_transitions = []
        
        for i in range(len(template_sequence)-1):
            template_transitions.append(f"{template_sequence[i]} → {template_sequence[i+1]}")
        
        # 예측 실패 원인 분석
        prediction_analysis = self._analyze_prediction_failure(
            sequence_context, target_log, predicted_template
        )
        
        # 컨텍스트 내 이상 패턴
        context_anomalies = self._detect_context_anomalies(sequence_context, target_log)
        
        return {
            'predicted_template': predicted_template,
            'actual_template': str(target_log['template_id']),
            'template_sequence': template_sequence,
            'recent_transitions': template_transitions[-3:] if len(template_transitions) >= 3 else template_transitions,
            'prediction_failure_reason': prediction_analysis,
            'context_anomalies': context_anomalies,
            'sequence_uniqueness': len(set(template_sequence)) / len(template_sequence) if template_sequence else 0
        }
    
    def _extract_template_anomaly_sample(self, anomaly: Dict, df_parsed: pd.DataFrame) -> Optional[Dict]:
        """템플릿 관련 이상에서 로그 샘플을 추출합니다."""
        
        # unique_templates 정보를 활용
        unique_templates = anomaly.get('unique_templates', [])
        if not unique_templates:
            return None
        
        # 고유 템플릿들의 로그 샘플 찾기
        sample_logs = []
        for template_id in unique_templates[:3]:  # 최대 3개
            template_logs = df_parsed[df_parsed['template_id'] == template_id]
            if len(template_logs) > 0:
                # 첫 번째와 마지막 로그 (시간 범위 파악용)
                first_log = template_logs.iloc[0]
                last_log = template_logs.iloc[-1] if len(template_logs) > 1 else first_log
                
                sample_logs.append({
                    'template_id': str(template_id),
                    'count': len(template_logs),
                    'first_occurrence': {
                        'line_no': int(first_log['line_no']),
                        'timestamp': str(first_log['timestamp']) if first_log['timestamp'] else None,
                        'raw_message': str(first_log['raw']),
                        'template': str(first_log['template'])
                    },
                    'last_occurrence': {
                        'line_no': int(last_log['line_no']),
                        'timestamp': str(last_log['timestamp']) if last_log['timestamp'] else None,
                        'raw_message': str(last_log['raw'])
                    } if len(template_logs) > 1 else None
                })
        
        return {
            'anomaly_type': anomaly.get('type', 'template_anomaly'),
            'severity': anomaly.get('severity', 'unknown'),
            'description': anomaly.get('description', 'Template distribution anomaly'),
            'unique_template_samples': sample_logs,
            'analysis': {
                'total_unique_templates': len(unique_templates),
                'kl_divergence': anomaly.get('kl_divergence'),
                'anomaly_explanation': self._explain_template_anomaly(anomaly)
            }
        }
    
    def _extract_metric_anomaly_sample(self, anomaly: Dict, df_parsed: pd.DataFrame) -> Optional[Dict]:
        """메트릭 관련 이상에서 로그 샘플을 추출합니다."""
        
        target_value = anomaly.get('target_value')
        baseline_mean = anomaly.get('baseline_mean')
        z_score = anomaly.get('z_score')
        
        # 전체 파일에서 대표적인 로그 샘플들 추출
        total_logs = len(df_parsed)
        sample_indices = np.linspace(0, total_logs-1, min(5, total_logs)).astype(int)
        sample_logs = []
        
        for idx in sample_indices:
            log_entry = df_parsed.iloc[idx]
            sample_logs.append({
                'line_no': int(log_entry['line_no']),
                'timestamp': str(log_entry['timestamp']) if log_entry['timestamp'] else None,
                'raw_message': str(log_entry['raw']),
                'template': str(log_entry['template']),
                'template_id': str(log_entry['template_id'])
            })
        
        return {
            'anomaly_type': anomaly.get('type', 'metric_anomaly'),
            'severity': anomaly.get('severity', 'unknown'),
            'description': anomaly.get('description', 'Metric anomaly'),
            'metric_comparison': {
                'target_value': target_value,
                'baseline_mean': baseline_mean,
                'z_score': z_score,
                'deviation_percentage': abs(target_value - baseline_mean) / max(baseline_mean, 1e-6) * 100 if target_value and baseline_mean else None
            },
            'representative_logs': sample_logs,
            'analysis': {
                'anomaly_explanation': self._explain_metric_anomaly(anomaly)
            }
        }
    
    def _extract_general_anomaly_sample(self, anomaly: Dict, df_parsed: pd.DataFrame) -> Optional[Dict]:
        """일반적인 이상에서 로그 샘플을 추출합니다."""
        
        # 무작위로 몇 개 로그 샘플 추출
        sample_size = min(3, len(df_parsed))
        sample_logs = df_parsed.sample(n=sample_size)
        
        return {
            'anomaly_type': anomaly.get('type', 'general_anomaly'),
            'severity': anomaly.get('severity', 'unknown'),
            'description': anomaly.get('description', 'General anomaly detected'),
            'sample_logs': self._format_log_entries(sample_logs),
            'analysis': {
                'anomaly_explanation': f"일반적인 이상 패턴: {anomaly.get('description', 'Unknown anomaly')}"
            }
        }
    
    def _format_log_entries(self, logs_df: pd.DataFrame) -> List[Dict]:
        """로그 엔트리들을 표준 형식으로 포맷합니다."""
        
        formatted_logs = []
        for _, log in logs_df.iterrows():
            formatted_logs.append({
                'line_no': int(log['line_no']),
                'timestamp': str(log['timestamp']) if log['timestamp'] else None,
                'host': str(log['host']) if 'host' in log and log['host'] else None,
                'process': str(log['process']) if 'process' in log and log['process'] else None,
                'raw_message': str(log['raw']),
                'masked_message': str(log['masked']) if 'masked' in log else None,
                'template': str(log['template']) if 'template' in log else None,
                'template_id': str(log['template_id']) if 'template_id' in log else None
            })
        
        return formatted_logs
    
    def _select_representative_logs(self, window_logs: pd.DataFrame, analysis: Dict) -> List[Dict]:
        """윈도우에서 대표적인 로그들을 선별합니다."""
        
        representatives = []
        
        # 1. 가장 많이 나온 템플릿의 첫 번째 로그
        dominant_templates = analysis.get('dominant_templates', [])
        if dominant_templates:
            top_template_id = dominant_templates[0]['template_id']
            top_template_logs = window_logs[window_logs['template_id'] == top_template_id]
            if len(top_template_logs) > 0:
                representatives.append({
                    'type': 'dominant_template',
                    'log': self._format_log_entries(top_template_logs.head(1))[0],
                    'reason': f"가장 빈번한 템플릿 (전체의 {dominant_templates[0]['percentage']:.1f}%)"
                })
        
        # 2. 에러 로그가 있다면 첫 번째 에러
        error_keywords = ['error', 'ERROR', 'fail', 'FAIL', 'exception', 'Exception']
        error_logs = window_logs[window_logs['raw'].str.contains('|'.join(error_keywords), case=False, na=False)]
        if len(error_logs) > 0:
            representatives.append({
                'type': 'error_log',
                'log': self._format_log_entries(error_logs.head(1))[0],
                'reason': "에러 메시지 포함"
            })
        
        # 3. 희귀 템플릿 (1번만 나온 것)
        template_counts = window_logs['template_id'].value_counts()
        rare_templates = template_counts[template_counts == 1]
        if len(rare_templates) > 0:
            rare_template_id = rare_templates.index[0]
            rare_log = window_logs[window_logs['template_id'] == rare_template_id].iloc[0]
            representatives.append({
                'type': 'rare_template',
                'log': self._format_log_entries(pd.DataFrame([rare_log]))[0],
                'reason': "새로운/희귀 템플릿"
            })
        
        return representatives[:3]  # 최대 3개
    
    def _analyze_time_density(self, logs_df: pd.DataFrame) -> Dict:
        """시간 밀도를 분석합니다."""
        
        if 'timestamp' not in logs_df.columns or len(logs_df) < 2:
            return {'analysis': 'insufficient_data'}
        
        try:
            logs_df = logs_df.copy()
            logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'], errors='coerce')
            valid_logs = logs_df.dropna(subset=['timestamp']).sort_values('timestamp')
            
            if len(valid_logs) < 2:
                return {'analysis': 'insufficient_valid_timestamps'}
            
            time_diffs = valid_logs['timestamp'].diff().dt.total_seconds().dropna()
            
            return {
                'total_duration_seconds': (valid_logs['timestamp'].max() - valid_logs['timestamp'].min()).total_seconds(),
                'average_interval_seconds': float(time_diffs.mean()),
                'logs_per_second': len(valid_logs) / max((valid_logs['timestamp'].max() - valid_logs['timestamp'].min()).total_seconds(), 1),
                'time_distribution': 'concentrated' if time_diffs.std() < time_diffs.mean() * 0.5 else 'spread'
            }
        except (ValueError, TypeError, pd.errors.ParserError):
            return {'analysis': 'time_analysis_failed'}
    
    def _identify_anomaly_indicators(self, window_logs: pd.DataFrame, unseen_rate: float, 
                                   freq_z: float, error_logs: pd.DataFrame, warning_logs: pd.DataFrame) -> List[str]:
        """이상 지표들을 식별합니다."""
        
        indicators = []
        
        if unseen_rate > 0.2:
            indicators.append(f"새로운 템플릿 비율이 높음 ({unseen_rate:.1%})")
        
        if freq_z > 2.0:
            indicators.append(f"템플릿 빈도 급증 (Z-score: {freq_z:.2f})")
        
        if len(error_logs) > 0:
            indicators.append(f"에러 로그 {len(error_logs)}개 발견")
        
        if len(warning_logs) > len(window_logs) * 0.1:
            indicators.append(f"경고 로그가 전체의 {len(warning_logs)/len(window_logs)*100:.1f}%")
        
        # 템플릿 다양성 체크
        unique_templates = len(window_logs['template_id'].unique())
        if unique_templates > len(window_logs) * 0.8:
            indicators.append("템플릿 다양성이 매우 높음 (거의 모든 로그가 다른 패턴)")
        
        return indicators
    
    def _analyze_prediction_failure(self, sequence_context: pd.DataFrame, target_log: pd.Series, 
                                  predicted_template: str) -> str:
        """예측 실패 원인을 분석합니다."""
        
        actual_template = str(target_log['template_id'])
        
        # 컨텍스트에서 해당 템플릿이 나온 적이 있는지 확인
        if actual_template in sequence_context['template_id'].values:
            recent_occurrence = sequence_context[sequence_context['template_id'] == actual_template].iloc[-1]
            position = len(sequence_context) - list(sequence_context['template_id']).index(actual_template)
            return f"예측하지 못한 템플릿 재등장 ({position}번째 전에 마지막 출현)"
        else:
            return f"컨텍스트에 없던 완전히 새로운 템플릿 등장"
    
    def _detect_context_anomalies(self, sequence_context: pd.DataFrame, target_log: pd.Series) -> List[str]:
        """컨텍스트 내 이상 패턴을 탐지합니다."""
        
        anomalies = []
        
        if len(sequence_context) == 0:
            return ["컨텍스트 부족"]
        
        # 1. 반복 패턴 체크
        templates = sequence_context['template_id'].tolist()
        if len(templates) > 3:
            last_3 = templates[-3:]
            if len(set(last_3)) == 1:
                anomalies.append("직전 3개 로그가 모두 동일한 템플릿")
        
        # 2. 에러 패턴 체크
        error_keywords = ['error', 'ERROR', 'fail', 'FAIL']
        error_count = sum(1 for _, log in sequence_context.iterrows() 
                         if any(keyword in str(log['raw']) for keyword in error_keywords))
        
        if error_count > len(sequence_context) * 0.3:
            anomalies.append(f"컨텍스트 내 에러 로그 비율이 높음 ({error_count}/{len(sequence_context)})")
        
        # 3. 시간 간격 체크 (만약 timestamp가 있다면)
        if 'timestamp' in sequence_context.columns:
            try:
                sequence_context = sequence_context.copy()
                sequence_context['timestamp'] = pd.to_datetime(sequence_context['timestamp'], errors='coerce')
                valid_times = sequence_context.dropna(subset=['timestamp'])
                
                if len(valid_times) > 1:
                    time_diffs = valid_times['timestamp'].diff().dt.total_seconds().dropna()
                    if time_diffs.max() > 3600:  # 1시간 이상 갭
                        anomalies.append("컨텍스트 내 큰 시간 간격 존재")
            except (ValueError, TypeError, pd.errors.ParserError):
                pass
        
        return anomalies if anomalies else ["특별한 컨텍스트 이상 없음"]
    
    def _explain_template_anomaly(self, anomaly: Dict) -> str:
        """템플릿 이상에 대한 설명을 생성합니다."""
        
        kl_div = anomaly.get('kl_divergence', 0)
        unique_count = anomaly.get('unique_templates', 0)
        
        explanation = f"이 파일은 다른 baseline 파일들과 비교했을 때 "
        
        if kl_div > 2.0:
            explanation += "템플릿 분포가 매우 다릅니다. "
        elif kl_div > 1.0:
            explanation += "템플릿 분포가 상당히 다릅니다. "
        
        if unique_count > 0:
            explanation += f"특히 {unique_count}개의 고유한 템플릿이 이 파일에만 나타납니다. "
            explanation += "이는 새로운 종류의 이벤트나 문제 상황을 나타낼 수 있습니다."
        
        return explanation
    
    def _explain_metric_anomaly(self, anomaly: Dict) -> str:
        """메트릭 이상에 대한 설명을 생성합니다."""
        
        metric_type = anomaly.get('type', '')
        target_value = anomaly.get('target_value')
        baseline_mean = anomaly.get('baseline_mean')
        
        if 'error_rate' in metric_type:
            if target_value > baseline_mean:
                return f"에러율이 baseline 평균 {baseline_mean:.2%}보다 {target_value:.2%}로 높습니다. 시스템에 문제가 있을 가능성이 있습니다."
            else:
                return f"에러율이 baseline 평균보다 낮습니다. 이는 양호한 상태일 수 있습니다."
        
        elif 'template' in metric_type:
            if target_value > baseline_mean:
                return f"템플릿 수가 baseline 평균 {baseline_mean:.0f}개보다 {target_value:.0f}개로 많습니다. 더 다양한 이벤트가 발생했을 수 있습니다."
            else:
                return f"템플릿 수가 baseline 평균보다 적습니다. 활동이 제한적이었을 수 있습니다."
        
        return f"메트릭 값이 baseline과 크게 다릅니다 (target: {target_value}, baseline: {baseline_mean})."

def analyze_all_anomalies(processed_dir: str, output_dir: str = None):
    """모든 이상탐지 결과에서 로그 샘플을 추출하고 분석합니다."""
    
    processed_path = Path(processed_dir)
    if output_dir is None:
        output_dir = processed_path / "log_samples_analysis"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🔍 이상 로그 샘플 분석 시작...")
    
    analyzer = LogSampleAnalyzer()
    all_results = {}
    
    # 필수 파일들 확인
    parsed_file = processed_path / "parsed.parquet"
    if not parsed_file.exists():
        print("❌ parsed.parquet 파일이 없습니다")
        return
    
    # 1. Baseline 이상탐지 결과 분석
    baseline_scores_file = processed_path / "baseline_scores.parquet"
    if baseline_scores_file.exists():
        print("📊 Baseline 이상탐지 결과 분석 중...")
        baseline_results = analyzer.extract_baseline_anomaly_samples(
            str(parsed_file), str(baseline_scores_file)
        )
        all_results['baseline'] = baseline_results
        print(f"   ✅ {baseline_results['anomaly_count']}개 이상 윈도우 중 {baseline_results['analyzed_count']}개 분석")
    
    # 2. DeepLog 이상탐지 결과 분석
    deeplog_infer_file = processed_path / "deeplog_infer.parquet"
    vocab_file = processed_path / "vocab.json"
    if deeplog_infer_file.exists() and vocab_file.exists():
        print("🧠 DeepLog 이상탐지 결과 분석 중...")
        deeplog_results = analyzer.extract_deeplog_anomaly_samples(
            str(parsed_file), str(deeplog_infer_file), str(vocab_file)
        )
        all_results['deeplog'] = deeplog_results
        print(f"   ✅ {deeplog_results['anomaly_count']}개 예측 실패 중 {deeplog_results['analyzed_count']}개 분석")
    
    # 3. 비교 분석 결과
    comparative_dir = processed_path / "comparative_analysis"
    comparative_anomalies_file = comparative_dir / "comparative_anomalies.json"
    if comparative_anomalies_file.exists():
        print("📊 비교 분석 이상탐지 결과 분석 중...")
        comparative_results = analyzer.extract_comparative_anomaly_samples(
            str(comparative_anomalies_file), str(parsed_file)
        )
        all_results['comparative'] = comparative_results
        print(f"   ✅ {comparative_results['anomaly_count']}개 비교 이상 중 {comparative_results['analyzed_count']}개 분석")
    
    # 결과 저장
    with open(output_path / "anomaly_samples.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    # 사람이 읽기 쉬운 리포트 생성
    print("📄 사람이 읽기 쉬운 리포트 생성 중...")
    report = generate_human_readable_report(all_results)
    with open(output_path / "anomaly_analysis_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 로그 샘플 분석 완료!")
    print(f"📂 결과 저장 위치: {output_path}")
    print(f"📄 주요 파일:")
    print(f"   - anomaly_samples.json: 상세 분석 데이터")
    print(f"   - anomaly_analysis_report.md: 사람이 읽기 쉬운 리포트")
    
    return all_results

def generate_human_readable_report(all_results: Dict) -> str:
    """사람이 읽기 쉬운 이상 로그 분석 리포트를 생성합니다."""
    
    report = f"""# 🔍 이상 로그 샘플 분석 리포트

**생성 시간**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

이 리포트는 다양한 이상탐지 방법으로 발견된 문제 로그들의 실제 샘플과 분석을 제공합니다.

## 📋 분석 방법별 요약

"""
    
    total_anomalies = 0
    for method, results in all_results.items():
        anomaly_count = results.get('anomaly_count', 0)
        total_anomalies += anomaly_count
        method_description = results.get('method', 'Unknown method')
        
        report += f"""### {method.title()} 방법
- **방법론**: {method_description}
- **발견된 이상**: {anomaly_count}개
- **분석된 샘플**: {results.get('analyzed_count', 0)}개

"""
    
    report += f"**전체 이상 개수**: {total_anomalies}개\n\n"
    
    # 각 방법별 상세 분석
    for method, results in all_results.items():
        if results.get('analyzed_count', 0) == 0:
            continue
            
        report += f"""## 🔬 {method.title()} 상세 분석

### 방법 설명
{results.get('method', 'Unknown method')}

"""
        
        samples = results.get('samples', [])
        for i, sample in enumerate(samples, 1):
            report += generate_sample_analysis(method, sample, i)
    
    report += """

## 💡 권장사항

### 즉시 조치가 필요한 사항
1. **에러 로그가 포함된 윈도우/시퀀스**: 즉시 원인 조사 필요
2. **새로운 템플릿 대량 발생**: 시스템 변경사항이나 새로운 문제 확인
3. **예측 불가능한 패턴**: 비정상적인 시스템 동작 가능성

### 추가 분석 방법
1. **시간대별 분석**: 특정 시간대에 집중된 이상이 있는지 확인
2. **호스트별 분석**: 특정 서버/호스트에서만 발생하는 문제인지 확인
3. **프로세스별 분석**: 특정 프로세스와 관련된 문제인지 확인

### 모니터링 강화
- 이상 패턴으로 식별된 템플릿들에 대한 알림 설정
- 에러율이 급증하는 구간에 대한 실시간 모니터링
- 새로운 템플릿 출현에 대한 자동 알림

---

**참고**: 이 리포트는 자동 생성되었습니다. 더 정확한 분석을 위해서는 시스템 관리자의 추가 검토가 필요합니다.
"""
    
    return report

def generate_sample_analysis(method: str, sample: Dict, sample_num: int) -> str:
    """개별 샘플에 대한 분석 섹션을 생성합니다."""
    
    if method == 'baseline':
        return generate_baseline_sample_analysis(sample, sample_num)
    elif method == 'deeplog':
        return generate_deeplog_sample_analysis(sample, sample_num)
    elif method == 'comparative':
        return generate_comparative_sample_analysis(sample, sample_num)
    else:
        return f"### 샘플 {sample_num}\n알 수 없는 분석 방법\n\n"

def generate_baseline_sample_analysis(sample: Dict, sample_num: int) -> str:
    """베이스라인 샘플 분석을 생성합니다."""
    
    window_start = sample['window_start_line']
    score = sample['anomaly_score']
    time_range = sample.get('time_range', {})
    analysis = sample.get('analysis', {})
    
    report = f"""### 🚨 이상 윈도우 #{sample_num} (라인 {window_start}~)

**기본 정보**:
- 이상 점수: {score:.3f}
- 새 템플릿 비율: {sample['unseen_rate']:.1%}
- 시간 범위: {time_range.get('start', 'N/A')} ~ {time_range.get('end', 'N/A')}
- 윈도우 내 로그 수: {sample['total_logs_in_window']}개

**이상 지표**:
"""
    
    indicators = analysis.get('anomaly_indicators', [])
    if indicators:
        for indicator in indicators:
            report += f"- ⚠️ {indicator}\n"
    else:
        report += "- 특별한 이상 지표 없음\n"
    
    # 대표 로그들
    representatives = sample.get('representative_logs', [])
    if representatives:
        report += "\n**대표적인 문제 로그들**:\n"
        for rep in representatives:
            log_info = rep['log']
            report += f"""
**{rep['type'].upper()}** ({rep['reason']}):
```
[{log_info.get('timestamp', 'N/A')}] {log_info.get('raw_message', 'N/A')}
```
- 템플릿: `{log_info.get('template', 'N/A')}`
- 라인: {log_info.get('line_no', 'N/A')}

"""
    
    # 전후 맥락
    context_before = sample.get('context_before', [])
    context_after = sample.get('context_after', [])
    
    if context_before or context_after:
        report += "**전후 맥락**:\n"
        
        if context_before:
            report += "```\n[이전 맥락]\n"
            for log in context_before[-2:]:  # 마지막 2개만
                report += f"{log.get('line_no', '?'):>6}: {log.get('raw_message', 'N/A')}\n"
            report += "```\n"
        
        if context_after:
            report += "```\n[이후 맥락]\n"
            for log in context_after[:2]:  # 처음 2개만
                report += f"{log.get('line_no', '?'):>6}: {log.get('raw_message', 'N/A')}\n"
            report += "```\n"
    
    report += "\n---\n\n"
    return report

def generate_deeplog_sample_analysis(sample: Dict, sample_num: int) -> str:
    """DeepLog 샘플 분석을 생성합니다."""
    
    target_log = sample.get('target_log', {})
    analysis = sample.get('analysis', {})
    
    report = f"""### 🧠 DeepLog 예측 실패 #{sample_num}

**기본 정보**:
- 시퀀스 인덱스: {sample['sequence_index']}
- 예측된 템플릿: `{sample['predicted_template_id']}`
- 실제 템플릿: `{sample['actual_template_id']}`
- 실제 라인 번호: {sample['actual_line_no']}

**실제 발생한 로그**:
```
[{target_log.get('timestamp', 'N/A')}] {target_log.get('raw_message', 'N/A')}
```
- 템플릿: `{target_log.get('template', 'N/A')}`

**예측 실패 원인**:
{analysis.get('prediction_failure_reason', '알 수 없음')}

**최근 템플릿 전환 패턴**:
"""
    
    recent_transitions = analysis.get('recent_transitions', [])
    if recent_transitions:
        for transition in recent_transitions:
            report += f"- `{transition}`\n"
    else:
        report += "- 패턴 정보 없음\n"
    
    # 컨텍스트 이상
    context_anomalies = analysis.get('context_anomalies', [])
    if context_anomalies:
        report += "\n**컨텍스트 분석**:\n"
        for anomaly in context_anomalies:
            report += f"- {anomaly}\n"
    
    # 시퀀스 컨텍스트
    sequence_context = sample.get('sequence_context', [])
    if sequence_context:
        report += "\n**시퀀스 컨텍스트 (최근 5개)**:\n```\n"
        for log in sequence_context[-5:]:
            report += f"{log.get('line_no', '?'):>6}: {log.get('raw_message', 'N/A')}\n"
        report += "```\n"
    
    report += "\n---\n\n"
    return report

def generate_comparative_sample_analysis(sample: Dict, sample_num: int) -> str:
    """비교 분석 샘플 분석을 생성합니다."""
    
    report = f"""### 📊 비교 분석 이상 #{sample_num}

**이상 유형**: {sample.get('anomaly_type', 'unknown')}
**심각도**: {sample.get('severity', 'unknown')}
**설명**: {sample.get('description', 'No description')}

"""
    
    # 템플릿 관련 이상
    if 'unique_template_samples' in sample:
        unique_samples = sample['unique_template_samples']
        report += "**고유 템플릿들**:\n"
        
        for template_sample in unique_samples:
            first_occ = template_sample['first_occurrence']
            report += f"""
**템플릿 ID**: `{template_sample['template_id']}`
**출현 횟수**: {template_sample['count']}회
**첫 번째 출현**:
```
[{first_occ.get('timestamp', 'N/A')}] {first_occ.get('raw_message', 'N/A')}
```
"""
    
    # 메트릭 관련 이상
    if 'metric_comparison' in sample:
        metric_comp = sample['metric_comparison']
        report += f"""
**메트릭 비교**:
- Target 값: {metric_comp.get('target_value', 'N/A')}
- Baseline 평균: {metric_comp.get('baseline_mean', 'N/A')}
- Z-Score: {metric_comp.get('z_score', 'N/A')}
- 편차: {metric_comp.get('deviation_percentage', 'N/A'):.1f}%
"""
    
    # 대표 로그들
    if 'representative_logs' in sample:
        rep_logs = sample['representative_logs']
        report += "\n**대표 로그 샘플들**:\n```\n"
        for log in rep_logs[:3]:
            report += f"{log.get('line_no', '?'):>6}: {log.get('raw_message', 'N/A')}\n"
        report += "```\n"
    
    # 분석 설명
    analysis = sample.get('analysis', {})
    explanation = analysis.get('anomaly_explanation', '')
    if explanation:
        report += f"\n**분석**: {explanation}\n"
    
    report += "\n---\n\n"
    return report

def main():
    parser = argparse.ArgumentParser(description="이상 로그 샘플 추출 및 분석")
    parser.add_argument("processed_dir", help="전처리된 결과 디렉토리")
    parser.add_argument("--output-dir", help="결과 출력 디렉토리")
    parser.add_argument("--max-samples", type=int, default=5, help="타입별 최대 샘플 수")
    parser.add_argument("--context-lines", type=int, default=3, help="전후 맥락 라인 수")
    
    args = parser.parse_args()
    
    # 글로벌 설정 업데이트
    LogSampleAnalyzer.max_samples_per_type = args.max_samples
    LogSampleAnalyzer.context_lines = args.context_lines
    
    analyze_all_anomalies(args.processed_dir, args.output_dir)

if __name__ == "__main__":
    main()
