"""
Logging utilities for nmr-spectra-processing.

Provides colored terminal output to replace R's crayon package.
Uses rich library for color formatting.
"""

from typing import Optional
from rich.console import Console


class Logger:
    """
    Simple logger with colored output.

    Replaces R package's crayon::yellow() and crayon::red() functions
    with colored terminal output.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize logger.

        Args:
            console: Optional rich Console instance. If None, creates a new one.
        """
        self.console = console or Console()

    def warning(self, message: str, prefix: str = ""):
        """
        Print a warning message in yellow.

        Replaces R's crayon::yellow() for warning messages.

        Args:
            message: Warning message to display
            prefix: Optional prefix (e.g., function name)
        """
        if prefix:
            self.console.print(f"[bold yellow]{prefix}[/bold yellow] [yellow]{message}[/yellow]")
        else:
            self.console.print(f"[yellow]{message}[/yellow]")

    def error(self, message: str, prefix: str = ""):
        """
        Print an error message in red.

        Replaces R's crayon::red() for error messages.

        Args:
            message: Error message to display
            prefix: Optional prefix (e.g., function name)
        """
        if prefix:
            self.console.print(f"[bold red]{prefix}[/bold red] [red]{message}[/red]")
        else:
            self.console.print(f"[red]{message}[/red]")

    def info(self, message: str, style: str = "blue"):
        """
        Print an info message.

        Args:
            message: Info message to display
            style: Rich style string (default: "blue")
        """
        self.console.print(f"[{style}]{message}[/{style}]")

    def success(self, message: str):
        """
        Print a success message in green.

        Args:
            message: Success message to display
        """
        self.console.print(f"[green]✓ {message}[/green]")


# Global logger instance
_default_logger: Optional[Logger] = None


def get_logger() -> Logger:
    """
    Get the default logger instance.

    Returns:
        Logger: Global logger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = Logger()
    return _default_logger


def warning(message: str, prefix: str = ""):
    """
    Print a warning message (convenience function).

    Args:
        message: Warning message
        prefix: Optional prefix
    """
    get_logger().warning(message, prefix)


def error(message: str, prefix: str = ""):
    """
    Print an error message (convenience function).

    Args:
        message: Error message
        prefix: Optional prefix
    """
    get_logger().error(message, prefix)


def info(message: str, style: str = "blue"):
    """
    Print an info message (convenience function).

    Args:
        message: Info message
        style: Rich style string
    """
    get_logger().info(message, style)


def success(message: str):
    """
    Print a success message (convenience function).

    Args:
        message: Success message
    """
    get_logger().success(message)
