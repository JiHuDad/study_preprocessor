#!/usr/bin/env python3
"""
테스트용 로그 생성기
정상 로그와 이상 로그를 체계적으로 생성합니다.
"""

import random
import datetime
from pathlib import Path

def generate_normal_web_logs(filename: str, count: int = 1000):
    """정상적인 웹서버 로그 생성"""
    ips = ['192.168.1.10', '10.0.0.5', '172.16.0.100', '192.168.0.50']
    methods = ['GET', 'POST', 'PUT', 'DELETE']
    urls = ['/api/users', '/api/products', '/dashboard', '/login', '/logout', '/home']
    status_codes = [200, 201, 204, 301, 304]  # 정상 상태 코드만
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    ]
    
    with open(filename, 'w') as f:
        start_time = datetime.datetime.now() - datetime.timedelta(hours=24)
        
        for i in range(count):
            timestamp = start_time + datetime.timedelta(seconds=i*3 + random.randint(0, 10))
            ip = random.choice(ips)
            method = random.choice(methods)
            url = random.choice(urls)
            status = random.choice(status_codes)
            size = random.randint(100, 5000)
            user_agent = random.choice(user_agents)
            
            log_line = f'{ip} - - [{timestamp.strftime("%d/%b/%Y:%H:%M:%S +0900")}] "{method} {url} HTTP/1.1" {status} {size} "-" "{user_agent}"\n'
            f.write(log_line)

def generate_normal_system_logs(filename: str, count: int = 1000):
    """정상적인 시스템 로그 생성"""
    services = ['sshd', 'systemd', 'kernel', 'NetworkManager', 'cron', 'dbus']
    normal_events = [
        'Service started successfully',
        'Connection established',
        'Authentication successful',
        'Task completed',
        'Configuration loaded',
        'Network interface up',
        'Process forked',
        'Memory allocated',
        'File system mounted',
        'Database connection opened'
    ]
    
    with open(filename, 'w') as f:
        start_time = datetime.datetime.now() - datetime.timedelta(hours=12)
        
        for i in range(count):
            timestamp = start_time + datetime.timedelta(seconds=i*2 + random.randint(0, 30))
            service = random.choice(services)
            pid = random.randint(1000, 9999)
            event = random.choice(normal_events)
            
            log_line = f'{timestamp.strftime("%Y-%m-%d %H:%M:%S")} hostname {service}[{pid}]: {event}\n'
            f.write(log_line)

def generate_normal_app_logs(filename: str, count: int = 1000):
    """정상적인 애플리케이션 로그 생성"""
    log_levels = ['INFO', 'DEBUG', 'WARN']  # ERROR는 제외
    components = ['UserService', 'OrderService', 'PaymentService', 'NotificationService']
    normal_messages = [
        'User login successful for user ID {}',
        'Order {} processed successfully',
        'Payment transaction {} completed',
        'Email sent to user {}',
        'Cache hit for key {}',
        'Database query executed in {}ms',
        'API response sent with status 200',
        'Session created for user {}',
        'Configuration updated',
        'Health check passed'
    ]
    
    with open(filename, 'w') as f:
        start_time = datetime.datetime.now() - datetime.timedelta(hours=6)
        
        for i in range(count):
            timestamp = start_time + datetime.timedelta(seconds=i + random.randint(0, 5))
            level = random.choice(log_levels)
            component = random.choice(components)
            message = random.choice(normal_messages)
            
            # 메시지에 따라 적절한 값 채우기
            if '{}' in message:
                if 'user' in message.lower():
                    value = f'user_{random.randint(1000, 9999)}'
                elif 'order' in message.lower():
                    value = f'ORD_{random.randint(100000, 999999)}'
                elif 'transaction' in message.lower():
                    value = f'TXN_{random.randint(100000, 999999)}'
                elif 'ms' in message:
                    value = str(random.randint(10, 500))
                else:
                    value = str(random.randint(1, 1000))
                message = message.format(value)
            
            log_line = f'{timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]} [{level}] {component}: {message}\n'
            f.write(log_line)

def generate_anomaly_logs(filename: str, base_count: int = 1000):
    """이상이 있는 로그 생성 (정상 로그 + 이상 패턴)"""
    
    # 먼저 정상 로그들을 생성
    normal_logs = []
    
    # 시스템 로그
    services = ['sshd', 'systemd', 'kernel', 'NetworkManager']
    normal_events = [
        'Service started successfully',
        'Connection established', 
        'Authentication successful',
        'Task completed'
    ]
    
    start_time = datetime.datetime.now() - datetime.timedelta(hours=8)
    
    # 정상 로그 80%
    for i in range(int(base_count * 0.8)):
        timestamp = start_time + datetime.timedelta(seconds=i*2)
        service = random.choice(services)
        pid = random.randint(1000, 9999)
        event = random.choice(normal_events)
        
        log_line = f'{timestamp.strftime("%Y-%m-%d %H:%M:%S")} hostname {service}[{pid}]: {event}\n'
        normal_logs.append((timestamp, log_line))
    
    # 이상 로그 20%
    anomaly_patterns = [
        'CRITICAL: Out of memory error',
        'ERROR: Connection timeout after 30 seconds',
        'FATAL: Database connection failed',
        'ERROR: Authentication failed for user admin (attempt 5)',
        'CRITICAL: Disk space exceeded 95%',
        'ERROR: SSL certificate expired',
        'FATAL: Segmentation fault in process',
        'ERROR: Failed to bind to port 80: Address already in use',
        'CRITICAL: Temperature threshold exceeded',
        'ERROR: Permission denied for /etc/shadow'
    ]
    
    # 이상 로그들 추가
    for i in range(int(base_count * 0.2)):
        timestamp = start_time + datetime.timedelta(seconds=len(normal_logs)*2 + i*5)
        service = random.choice(services)
        pid = random.randint(1000, 9999)
        error = random.choice(anomaly_patterns)
        
        log_line = f'{timestamp.strftime("%Y-%m-%d %H:%M:%S")} hostname {service}[{pid}]: {error}\n'
        normal_logs.append((timestamp, log_line))
    
    # 시간순 정렬
    normal_logs.sort(key=lambda x: x[0])
    
    # 파일에 쓰기
    with open(filename, 'w') as f:
        for _, log_line in normal_logs:
            f.write(log_line)

if __name__ == '__main__':
    # 정상 로그들 생성
    print("Generating normal logs...")
    generate_normal_web_logs('baseline_logs/web_server.log', 800)
    generate_normal_system_logs('baseline_logs/system_service.log', 800) 
    generate_normal_app_logs('baseline_logs/application.log', 800)
    
    # 이상 로그 생성 (테스트용)
    print("Generating anomaly logs...")
    generate_anomaly_logs('target_logs/system_with_errors.log', 1000)
    
    print("✅ Test logs generated successfully!")
    print("📁 Baseline logs (normal): baseline_logs/")
    print("📁 Target logs (with anomalies): target_logs/")
