"""
Global configuration loader and helper functions for the QA/AI Blockchain Framework.
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager that loads settings from YAML files and environment variables.
    Environment variables take precedence over YAML configuration.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_file: Path to YAML configuration file
        """
        self._config: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)
        
        # Load default configuration
        self._load_defaults()
        
        # Load from YAML file if provided
        if config_file and os.path.exists(config_file):
            self._load_yaml(config_file)
        
        # Override with environment variables
        self._load_env_vars()
    
    def _load_defaults(self):
        """Load default configuration values."""
        self._config.update({
            'blockchain': {
                'network_config_path': 'network.yaml',
                'channel_name': 'qahealthchannel',
                'chaincode_name': 'aiqa_cc',
                'org_name': 'HealthcareOrg',
                'user_name': 'admin',
                'connection_timeout': 30
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'reports': {
                'output_dir': 'reports',
                'format': 'json'
            },
            'simulation': {
                'num_samples': 100,
                'random_seed': 42
            }
        })
    
    def _load_yaml(self, config_file: str):
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
        """
        try:
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    self._merge_config(self._config, yaml_config)
                    self._logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            self._logger.error(f"Failed to load YAML config from {config_file}: {e}")
    
    def _load_env_vars(self):
        """Load configuration from environment variables with FRAMEWORK_ prefix."""
        env_mappings = {
            'FRAMEWORK_BLOCKCHAIN_CHANNEL': ('blockchain', 'channel_name'),
            'FRAMEWORK_BLOCKCHAIN_CHAINCODE': ('blockchain', 'chaincode_name'),
            'FRAMEWORK_BLOCKCHAIN_ORG': ('blockchain', 'org_name'),
            'FRAMEWORK_BLOCKCHAIN_USER': ('blockchain', 'user_name'),
            'FRAMEWORK_LOG_LEVEL': ('logging', 'level'),
            'FRAMEWORK_REPORTS_DIR': ('reports', 'output_dir'),
            'FRAMEWORK_SIMULATION_SAMPLES': ('simulation', 'num_samples'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                if section not in self._config:
                    self._config[section] = {}
                # Convert numeric values
                if key in ['num_samples', 'connection_timeout']:
                    try:
                        value = int(value)
                    except ValueError:
                        self._logger.warning(f"Invalid numeric value for {env_var}: {value}")
                        continue
                self._config[section][key] = value
                self._logger.debug(f"Set {section}.{key} = {value} from environment")
    
    def _merge_config(self, base: Dict, override: Dict):
        """
        Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Configuration key path (e.g., 'blockchain.channel_name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
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
            key_path: Configuration key path (e.g., 'blockchain.channel_name')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get complete configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()


# Global configuration instance
_global_config: Optional[Config] = None


def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_file: Path to configuration file (used only on first call)
        
    Returns:
        Configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config(config_file)
    return _global_config


def reload_config(config_file: Optional[str] = None):
    """
    Reload global configuration.
    
    Args:
        config_file: Path to configuration file
    """
    global _global_config
    _global_config = Config(config_file)