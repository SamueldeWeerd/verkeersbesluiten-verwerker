import pytest
from unittest.mock import patch, Mock
import time
from src.utils.http_client import RateLimitedClient

from src.config.settings import Settings, DirectorySettings, RateLimitSettings
from pathlib import Path

def get_test_settings():
    """Create settings instance for testing with no delays"""
    return Settings(
        rate_limit=RateLimitSettings(request_delay=0),
        directories=DirectorySettings(
            verkeersbesluiten=Path("./verkeersbesluiten"),
            afbeeldingen=Path("./afbeeldingen")
        )
    )

def test_successful_request():
    client = RateLimitedClient(settings=get_test_settings())
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        response = client.get('http://test.com')
        assert response.ok
        assert response.status_code == 200

def test_rate_limit_handling():
    client = RateLimitedClient(settings=get_test_settings())
    with patch('requests.get') as mock_get:
        # First request triggers rate limit
        rate_limit_response = Mock()
        rate_limit_response.ok = False
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {'Retry-After': '1'}
        
        # Second request succeeds
        success_response = Mock()
        success_response.ok = True
        success_response.status_code = 200
        
        mock_get.side_effect = [rate_limit_response, success_response]
        
        response = client.get('http://test.com')
        assert response.ok
        assert response.status_code == 200
        assert mock_get.call_count == 2  # Verify retry happened