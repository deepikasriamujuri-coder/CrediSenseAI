import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "default-academic-credi-secret-key-123")
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # Limit request size to 2MB
    JSON_SORT_KEYS = False
    
    # Model and metadata paths
    MODEL_PATH = os.environ.get("MODEL_PATH", "models/credit_risk_model.joblib")
    METADATA_PATH = os.environ.get("METADATA_PATH", "models/model_metadata.json")
    
    # Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    # Use fallback secrets for testing
    SECRET_KEY = "test-secret-key-456"

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = "WARNING"

config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig
}
