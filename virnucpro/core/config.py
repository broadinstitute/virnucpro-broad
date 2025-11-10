"""Configuration management for VirNucPro"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger('virnucpro.config')


@dataclass
class Config:
    """
    Application configuration container.

    Loads configuration from YAML file with support for
    CLI overrides and validation.
    """

    # Raw configuration dictionary
    _config: Dict[str, Any] = field(default_factory=dict)

    # Configuration file path
    config_file: Optional[Path] = None

    @classmethod
    def load(cls, config_file: Optional[Path] = None) -> 'Config':
        """
        Load configuration from file.

        Args:
            config_file: Path to YAML config file. If None, uses default.

        Returns:
            Config instance
        """
        # Determine config file
        if config_file is None:
            # Use default config
            default_config = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
            config_file = default_config
        else:
            config_file = Path(config_file)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        # Load YAML
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        logger.debug(f"Loaded configuration from {config_file}")

        return cls(_config=config_dict, config_file=config_file)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., "prediction.batch_size")
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config.get("prediction.batch_size")
            256
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config

        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set value
        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self._config.copy()

    def save(self, output_file: Path):
        """
        Save configuration to YAML file.

        Args:
            output_file: Output file path
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)

        logger.info(f"Saved configuration to {output_file}")
