"""
CORS middleware configuration.
"""

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


def get_cors_origins() -> List[str]:
    """
    Get allowed CORS origins from environment.
    Returns a list of allowed origins for the CORS middleware.
    """
    origins_str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in origins_str.split(",") if origin.strip()]


def get_cors_config() -> dict:
    """
    Get full CORS configuration for FastAPI middleware.

    Returns:
        dict: Configuration dictionary for CORSMiddleware
    """
    return {
        "allow_origins": get_cors_origins(),
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-API-Key",
        ],
        "expose_headers": ["X-Request-ID"],
        "max_age": 600,  # Cache preflight for 10 minutes
    }
