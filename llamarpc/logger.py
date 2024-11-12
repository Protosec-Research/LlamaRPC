from rich.console import Console
from rich.logging import RichHandler
import logging
import sys

# Create console instance
console = Console()

# Configure logging with Rich
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console,
        show_path=False,
        enable_link_path=False,
        markup=True,
        rich_tracebacks=True
    )]
)

# Create logger
logger = logging.getLogger("llamarpc") 