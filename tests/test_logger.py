"""Tests for logger utility."""

import pytest
from io import StringIO
from rich.console import Console
from nmr_spectra_processing.utils.logger import Logger, get_logger


def test_logger_creation():
    """Test logger can be created."""
    logger = Logger()
    assert logger is not None
    assert logger.console is not None


def test_logger_with_custom_console():
    """Test logger with custom console."""
    console = Console(file=StringIO(), force_terminal=True)
    logger = Logger(console=console)
    assert logger.console is console


def test_warning_message():
    """Test warning message formatting."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    logger = Logger(console=console)

    logger.warning("Test warning message")
    result = output.getvalue()
    assert "Test warning message" in result


def test_warning_with_prefix():
    """Test warning with prefix."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    logger = Logger(console=console)

    logger.warning("Test warning", prefix="test_function")
    result = output.getvalue()
    assert "test_function" in result
    assert "Test warning" in result


def test_error_message():
    """Test error message formatting."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    logger = Logger(console=console)

    logger.error("Test error message")
    result = output.getvalue()
    assert "Test error message" in result


def test_error_with_prefix():
    """Test error with prefix."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    logger = Logger(console=console)

    logger.error("Test error", prefix="test_function")
    result = output.getvalue()
    assert "test_function" in result
    assert "Test error" in result


def test_info_message():
    """Test info message."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    logger = Logger(console=console)

    logger.info("Test info message")
    result = output.getvalue()
    assert "Test info message" in result


def test_success_message():
    """Test success message."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    logger = Logger(console=console)

    logger.success("Test success message")
    result = output.getvalue()
    assert "Test success message" in result
    assert "✓" in result or "√" in result  # Check mark may vary


def test_get_logger_singleton():
    """Test get_logger returns singleton."""
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2


def test_convenience_functions():
    """Test convenience functions work."""
    from nmr_spectra_processing.utils.logger import warning, error, info, success

    # These should not raise errors
    # (output goes to default console, we just check they run)
    warning("Test warning")
    error("Test error")
    info("Test info")
    success("Test success")
