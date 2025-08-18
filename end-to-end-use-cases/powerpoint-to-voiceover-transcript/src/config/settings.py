"""Configuration management using YAML files."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Global config cache
_config_cache: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, looks for config.yaml in project root.

    Returns:
        Dictionary containing configuration settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    global _config_cache

    if config_path is None:
        # Look for config.yaml in project root (parent of src directory)
        current_dir = Path(__file__).parent.parent.parent
        config_file_path = current_dir / "config.yaml"
    else:
        config_file_path = Path(config_path)

    if not config_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    try:
        with open(str(config_file_path), 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Cache the config
        _config_cache = config

        # Load environment variables for API keys
        _load_env_variables(config)

        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_file_path}: {e}")


def get_config() -> Dict[str, Any]:
    """
    Get cached configuration. Loads from default location if not already loaded.

    Returns:
        Dictionary containing configuration settings
    """
    global _config_cache

    if _config_cache is None:
        _config_cache = load_config()

    return _config_cache


def _load_env_variables(config: Dict[str, Any]) -> None:
    """
    Load environment variables and add them to config.

    Args:
        config: Configuration dictionary to update
    """
    # Load API key from environment
    llama_api_key = os.getenv('LLAMA_API_KEY')
    if llama_api_key:
        if 'api' not in config:
            config['api'] = {}
        config['api']['llama_api_key'] = llama_api_key


def get_system_prompt() -> str:
    """
    Get the system prompt from configuration.

    Returns:
        System prompt string
    """
    config = get_config()
    return config.get('system_prompt', '')


def get_api_config() -> Dict[str, Any]:
    """
    Get API configuration settings.

    Returns:
        Dictionary containing API settings
    """
    config = get_config()
    return config.get('api', {})


def get_processing_config() -> Dict[str, Any]:
    """
    Get processing configuration settings.

    Returns:
        Dictionary containing processing settings
    """
    config = get_config()
    return config.get('processing', {})


def get_paths_config() -> Dict[str, Any]:
    """
    Get file paths configuration.

    Returns:
        Dictionary containing path settings
    """
    config = get_config()
    return config.get('paths', {})


def get_libreoffice_paths() -> list:
    """
    Get possible LibreOffice installation paths.

    Returns:
        List of possible LibreOffice paths
    """
    config = get_config()
    return config.get('libreoffice', {}).get('possible_paths', [])


def get_image_quality_config() -> Dict[str, Any]:
    """
    Get image quality configuration settings.

    Returns:
        Dictionary containing image quality settings
    """
    config = get_config()
    return config.get('image_quality', {})


def get_knowledge_config() -> Dict[str, Any]:
    """
    Get knowledge base configuration settings.

    Returns:
        Dictionary containing knowledge base settings
    """
    config = get_config()
    return config.get('knowledge', {})


def is_knowledge_enabled() -> bool:
    """
    Check if knowledge base integration is enabled.

    Returns:
        True if knowledge base is enabled, False otherwise
    """
    knowledge_config = get_knowledge_config()
    return knowledge_config.get('enabled', False)


def validate_knowledge_config() -> None:
    """
    Validate knowledge base configuration parameters.

    Raises:
        ValueError: If configuration is invalid
    """
    from ..knowledge.exceptions import KnowledgeConfigurationError

    if not is_knowledge_enabled():
        return

    knowledge_config = get_knowledge_config()

    # Validate required fields
    required_fields = ['knowledge_base_dir', 'embedding', 'search', 'context']
    for field in required_fields:
        if field not in knowledge_config:
            raise KnowledgeConfigurationError(
                f"Missing required knowledge configuration field: {field}",
                config_key=field
            )

    # Validate embedding config
    embedding_config = knowledge_config.get('embedding', {})
    if 'model_name' not in embedding_config:
        raise KnowledgeConfigurationError(
            "Missing embedding model_name in knowledge configuration",
            config_key='embedding.model_name'
        )

    # Validate search config
    search_config = knowledge_config.get('search', {})
    top_k = search_config.get('top_k', 5)
    if not isinstance(top_k, int) or top_k <= 0:
        raise KnowledgeConfigurationError(
            f"Invalid top_k value: {top_k}. Must be a positive integer.",
            config_key='search.top_k',
            config_value=top_k
        )

    similarity_threshold = search_config.get('similarity_threshold', 0.3)
    if not isinstance(similarity_threshold, (int, float)) or not 0.0 <= similarity_threshold <= 1.0:
        raise KnowledgeConfigurationError(
            f"Invalid similarity_threshold: {similarity_threshold}. Must be between 0.0 and 1.0.",
            config_key='search.similarity_threshold',
            config_value=similarity_threshold
        )

    # Validate context config
    context_config = knowledge_config.get('context', {})
    strategy = context_config.get('strategy', 'combined')
    valid_strategies = ['knowledge_only', 'narrative_priority', 'combined']
    if strategy not in valid_strategies:
        raise KnowledgeConfigurationError(
            f"Invalid context strategy: {strategy}. Must be one of {valid_strategies}.",
            config_key='context.strategy',
            config_value=strategy
        )

    integration_method = context_config.get('integration_method', 'system_prompt')
    valid_methods = ['system_prompt', 'user_message']
    if integration_method not in valid_methods:
        raise KnowledgeConfigurationError(
            f"Invalid integration method: {integration_method}. Must be one of {valid_methods}.",
            config_key='context.integration_method',
            config_value=integration_method
        )
