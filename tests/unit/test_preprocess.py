"""
Unit tests for preprocess module
"""
import pytest
from anomaly_log_detector.preprocess import (
    mask_message,
    parse_line,
    PreprocessConfig,
    LogPreprocessor
)


class TestMaskMessage:
    """Tests for mask_message function"""

    def test_mask_hex_addresses(self):
        """Test hexadecimal address masking"""
        msg = "Allocated memory at 0x7f8a3c000000"
        result = mask_message(msg)
        assert "<HEX>" in result
        assert "0x7f8a3c000000" not in result

    def test_mask_ipv4(self):
        """Test IPv4 address masking"""
        msg = "Connection from 192.168.1.100"
        result = mask_message(msg)
        assert "<IP>" in result
        assert "192.168.1.100" not in result

    def test_mask_paths(self):
        """Test filesystem path masking"""
        msg = "Reading file /var/log/syslog"
        result = mask_message(msg)
        assert "<PATH>" in result
        assert "/var/log/syslog" not in result

    def test_mask_numbers(self):
        """Test numeric value masking"""
        msg = "Process used 1234 KB of memory"
        result = mask_message(msg)
        assert "<NUM>" in result

    def test_no_masking_when_disabled(self):
        """Test that masking can be disabled"""
        msg = "Value is 123"
        cfg = PreprocessConfig(mask_numbers=False)
        result = mask_message(msg, cfg)
        assert "123" in result


class TestParseLine:
    """Tests for parse_line function"""

    def test_parse_syslog_format(self):
        """Test parsing of syslog format"""
        line = "Sep 14 05:04:41 hostname kernel: Test message"
        ts, host, proc, msg = parse_line(line)
        assert host == "hostname"
        assert proc == "kernel"
        assert msg == "Test message"

    def test_parse_dmesg_format(self):
        """Test parsing of dmesg format"""
        line = "[  123.456789] Test kernel message"
        ts, host, proc, msg = parse_line(line)
        assert msg == "Test kernel message"
        assert host is None

    def test_parse_raw_line(self):
        """Test parsing of unformatted line"""
        line = "Just a plain log message"
        ts, host, proc, msg = parse_line(line)
        assert msg == "Just a plain log message"


class TestPreprocessConfig:
    """Tests for PreprocessConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        cfg = PreprocessConfig()
        assert cfg.mask_paths is True
        assert cfg.mask_hex is True
        assert cfg.mask_ips is True
        assert cfg.mask_numbers is True

    def test_custom_config(self):
        """Test custom configuration"""
        cfg = PreprocessConfig(
            mask_numbers=False,
            drain_state_path="/tmp/drain3.json"
        )
        assert cfg.mask_numbers is False
        assert cfg.drain_state_path == "/tmp/drain3.json"


class TestLogPreprocessor:
    """Tests for LogPreprocessor class"""

    def test_preprocessor_creation(self):
        """Test creating a preprocessor instance"""
        cfg = PreprocessConfig()
        preprocessor = LogPreprocessor(cfg)
        assert preprocessor is not None

    # Note: More tests would require mocking Drain3 or using test fixtures
    # This is a basic structure to get started
