import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management for the application"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    
    # System Configuration
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Agent Configuration
    AGENT_CONFIGS = {
        "planner": {
            "temperature": 0.2,
            "max_consecutive_auto_reply": 10
        },
        "coder": {
            "temperature": 0.1,
            "max_consecutive_auto_reply": 5
        },
        "debugger": {
            "temperature": 0.1,
            "max_consecutive_auto_reply": 3
        },
        "tester": {
            "temperature": 0.1,
            "max_consecutive_auto_reply": 5
        },
        "executor": {
            "temperature": 0.1,
            "max_consecutive_auto_reply": 3
        }
    }
    
    # LLM Default Configuration
    DEFAULT_LLM_CONFIG = {
        "timeout": 600,
        "retry_on_timeout": True,
        "max_retries": 3,
        "cache_seed": 42,
        "temperature": OPENAI_TEMPERATURE,
        "config_list": [{
            "model": OPENAI_MODEL,
            "api_key": OPENAI_API_KEY
        }]
    }
    
    @classmethod
    def get_openai_config(cls) -> Dict[str, Any]:
        """
        Get base OpenAI configuration.
        
        Returns:
            Dict containing OpenAI configuration
        """
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        return cls.DEFAULT_LLM_CONFIG
    
    @classmethod
    def get_agent_config(cls, agent_type: str) -> Dict[str, Any]:
        """
        Get agent-specific configuration.
        
        Args:
            agent_type: Type of agent (planner, coder, debugger, etc.)
            
        Returns:
            Dict containing agent configuration merged with base config
        """
        base_config = cls.get_openai_config()
        agent_config = cls.AGENT_CONFIGS.get(agent_type, {})
        
        return {
            **base_config,
            **agent_config
        }
    
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """
        Get logging configuration.
        
        Returns:
            Dict containing logging configuration
        """
        return {
            "level": cls.LOG_LEVEL,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "debug_mode": cls.DEBUG_MODE
        }
    
    @classmethod
    def validate_config(cls) -> None:
        """
        Validate the configuration.
        
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        required_vars = [
            ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
            ("OPENAI_MODEL", cls.OPENAI_MODEL)
        ]
        
        for var_name, var_value in required_vars:
            if not var_value:
                raise ValueError(f"Required configuration {var_name} is missing")
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize and validate configuration"""
        cls.validate_config()
        
        if cls.DEBUG_MODE:
            print("Running in DEBUG mode")
            print(f"Using OpenAI Model: {cls.OPENAI_MODEL}")
            print(f"Log Level: {cls.LOG_LEVEL}")

# Initialize configuration when module is imported
Config.initialize() 