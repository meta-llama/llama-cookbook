"""File system utilities for PPTX to Transcript."""

import shutil
from pathlib import Path
from ..config.settings import get_libreoffice_paths


def check_libreoffice() -> str:
    """
    Find LibreOffice soffice binary.

    Returns:
        Path to the soffice binary as a string

    Raises:
        FileNotFoundError: If LibreOffice is not found
    """
    # Get possible paths from config
    possible_paths = get_libreoffice_paths()

    # Add shutil.which result to the list
    which_result = shutil.which("soffice")
    if which_result:
        possible_paths.insert(1, which_result)  # Insert after the first path

    for path in possible_paths:
        if path and Path(path).exists():
            return str(path)

    raise FileNotFoundError(
        "LibreOffice not found! Please install LibreOffice or update the paths in config.yaml"
    )


def ensure_directory_exists(directory_path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory
    """
    directory_path.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to the project root directory
    """
    # Go up from src/core to project root
    return Path(__file__).parent.parent.parent
