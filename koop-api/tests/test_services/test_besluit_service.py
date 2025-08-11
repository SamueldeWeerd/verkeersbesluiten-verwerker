import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.services.besluit_download_service import BesluitService
from src.config.settings import Settings, DirectorySettings
from pathlib import Path

@pytest.fixture
def mock_settings():
    return Settings(
        directories=DirectorySettings(
            verkeersbesluiten=Path("./verkeersbesluiten"),
            afbeeldingen=Path("./afbeeldingen")
        )
    )

@pytest.fixture
def mock_http_client():
    return Mock()

@pytest.fixture
def mock_xml_parser():
    return Mock()

@pytest.fixture
def mock_image_classifier():
    return Mock()

@pytest.fixture
def service(mock_settings, mock_http_client, mock_xml_parser, mock_image_classifier):
    return BesluitService(
        settings=mock_settings,
        http_client=mock_http_client,
        xml_parser=mock_xml_parser,
        image_classifier=mock_image_classifier
    )

def test_invalid_date_format(service):
    with pytest.raises(ValueError, match="Date must be in YYYY-MM-DD format"):
        service.get_besluiten_for_date("invalid-date")

def test_successful_response(service, mock_http_client, mock_xml_parser):
    # Mock successful SRU response
    mock_response = Mock()
    mock_response.ok = True
    mock_response.content = b"<xml>test</xml>"
    mock_http_client.get.return_value = mock_response
    
    # Mock record parsing
    mock_record = Mock()
    mock_xml_parser.parse_sru_response.return_value = [mock_record]
    
    # Mock URL extraction
    mock_xml_parser.extract_urls_from_record.return_value = {
        "content": "http://example.com/content.xml",
        "metadata": "http://example.com/metadata.xml"
    }
    
    # Mock content processing
    mock_xml_parser.extract_plain_text.return_value = "Test content"
    mock_xml_parser.parse_metadata_block.return_value = {
        "OVERHEIDop.externeBijlage": "exb-2024-12345"
    }
    mock_xml_parser.extract_exb_code.return_value = "exb-2024-12345"
    mock_xml_parser.extract_embedded_images.return_value = ["image1.jpg"]
    
    result = service.get_besluiten_for_date("2024-01-01")
    
    assert len(result) == 1
    assert result[0]["text"] == "Test content"
    assert "exb-2024-12345" in result[0]["metadata"]["OVERHEIDop.externeBijlage"]

def test_no_records_found(service, mock_http_client, mock_xml_parser):
    mock_response = Mock()
    mock_response.ok = True
    mock_response.content = b"<xml>test</xml>"
    mock_http_client.get.return_value = mock_response
    
    # Mock empty records list
    mock_xml_parser.parse_sru_response.return_value = []
    
    result = service.get_besluiten_for_date("2024-01-01")
    assert result == []

def test_failed_sru_request(service, mock_http_client):
    mock_response = Mock()
    mock_response.ok = False
    mock_http_client.get.return_value = mock_response
    
    result = service.get_besluiten_for_date("2024-01-01")
    assert result == []

def test_excluded_content(service, mock_http_client, mock_xml_parser):
    # Mock successful SRU response
    mock_response = Mock()
    mock_response.ok = True
    mock_response.content = b"<xml>test</xml>"
    mock_http_client.get.return_value = mock_response
    
    # Mock record with excluded content
    mock_record = Mock()
    mock_xml_parser.parse_sru_response.return_value = [mock_record]
    mock_xml_parser.extract_urls_from_record.return_value = {
        "content": "http://example.com/content.xml"
    }
    
    # Set up content to contain excluded keyword
    service._settings.exclude_keywords = ["parkeerplaats"]
    mock_response_content = Mock()
    mock_response_content.ok = True
    mock_response_content.content = b"test parkeerplaats content"
    mock_http_client.get.return_value = mock_response_content
    
    result = service.get_besluiten_for_date("2024-01-01")
    assert result == []