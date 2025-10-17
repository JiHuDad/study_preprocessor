"""
Pytest configuration and shared fixtures
"""
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def tmp_log_file(tmp_path):
    """Create a temporary log file for testing"""
    log_file = tmp_path / "test.log"
    log_content = """Sep 14 05:04:41 hostname kernel: Test message 1
Sep 14 05:04:42 hostname sshd: Connection from 192.168.1.100
Sep 14 05:04:43 hostname kernel: Allocated memory at 0x7f8a3c000000
[  123.456789] dmesg style message
Plain log message without format
"""
    log_file.write_text(log_content)
    return log_file


@pytest.fixture
def sample_parsed_data():
    """Provide sample parsed log data for testing"""
    import pandas as pd
    return pd.DataFrame({
        'line_no': [1, 2, 3, 4, 5],
        'timestamp': [None, None, None, None, None],
        'host': ['host1', 'host1', 'host2', 'host1', 'host2'],
        'process': ['kernel', 'sshd', 'kernel', 'systemd', 'kernel'],
        'raw': ['msg1', 'msg2', 'msg3', 'msg4', 'msg5'],
        'masked': ['<MSG>', '<MSG>', '<MSG>', '<MSG>', '<MSG>'],
        'template_id': ['T1', 'T2', 'T1', 'T3', 'T1'],
        'template': ['Template 1', 'Template 2', 'Template 1', 'Template 3', 'Template 1']
    })


@pytest.fixture
def sample_vocab():
    """Provide sample vocabulary for DeepLog testing"""
    return {
        'T1': 0,
        'T2': 1,
        'T3': 2
    }


@pytest.fixture
def config_dir():
    """Provide path to config directory"""
    return Path(__file__).parent.parent / "config"


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def clean_cache(tmp_path):
    """Provide a clean cache directory"""
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    yield cache_dir
    # Cleanup after test
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


# Pytest hooks for custom test behavior
def pytest_configure(config):
    """Configure pytest with custom settings"""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers"""
    # Skip GPU tests if no GPU available
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    if not has_gpu:
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "requires_gpu" in item.keywords:
                item.add_marker(skip_gpu)
