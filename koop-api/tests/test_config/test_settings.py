import pytest
from datetime import datetime
from pathlib import Path
from src.config.settings import Settings, DirectorySettings

def test_api_base_url():
    settings = Settings(
        directories=DirectorySettings(
            verkeersbesluiten=Path("./verkeersbesluiten"),
            afbeeldingen=Path("./afbeeldingen")
        )
    )
    assert settings.api.base_url == "http://localhost:8001"

def test_date_validation():
    with pytest.raises(ValueError, match="Date must be in YYYY-MM-DD format"):
        Settings(
            date_range={"start": "invalid-date"},
            directories=DirectorySettings(
                verkeersbesluiten=Path("./verkeersbesluiten"),
                afbeeldingen=Path("./afbeeldingen")
            )
        )

def test_directory_creation(tmp_path):
    verkeer_dir = tmp_path / "verkeersbesluiten"
    afb_dir = tmp_path / "afbeeldingen"
    
    settings = Settings(
        directories=DirectorySettings(
            verkeersbesluiten=verkeer_dir,
            afbeeldingen=afb_dir
        )
    )
    
    assert verkeer_dir.exists()
    assert afb_dir.exists()

def test_environment_variables(monkeypatch):
    monkeypatch.setenv("VERKEERSBESLUIT_API__HOST", "testhost")
    monkeypatch.setenv("VERKEERSBESLUIT_API__PORT", "9000")
    
    settings = Settings(
        directories=DirectorySettings(
            verkeersbesluiten=Path("./verkeersbesluiten"),
            afbeeldingen=Path("./afbeeldingen")
        )
    )
    
    assert settings.api.host == "testhost"
    assert settings.api.port == 9000
    assert settings.api.base_url == "http://testhost:9000"