from datetime import datetime
from pathlib import Path
from typing import List
from pydantic import BaseModel, HttpUrl, validator
from pydantic_settings import BaseSettings
from functools import lru_cache

class APISettings(BaseModel):
    """API server configuration."""
    host: str = "localhost"
    port: int = 8000
    protocol: str = "http"
    # External service name for Docker network access (e.g., from N8N)
    external_service_name: str = "koop-api-service"
    # External base URL for browser/external access
    external_base_url_override: str = ""

    @property
    def base_url(self) -> str:
        """Constructs the base URL for the API service."""
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def external_base_url(self) -> str:
        """Constructs the external base URL for external access."""
        if self.external_base_url_override:
            return self.external_base_url_override
        return f"{self.protocol}://{self.external_service_name}:{self.port}"

class DateRangeSettings(BaseModel):
    """Date range configuration for API queries."""
    start: str = "2022-01-01"
    end: str = "2024-12-01"

    @validator("start", "end")
    def validate_date_format(cls, v: str) -> str:
        """Ensures dates are in YYYY-MM-DD format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

class DirectorySettings(BaseModel):
    """Directory paths configuration."""
    verkeersbesluiten: Path
    afbeeldingen: Path

    @validator("verkeersbesluiten", "afbeeldingen")
    def create_directory(cls, v: Path) -> Path:
        """Ensures directories exist, creates them if they don't."""
        v.mkdir(exist_ok=True)
        return v

class SRUSettings(BaseModel):
    """SRU API configuration."""
    base_url: HttpUrl = "https://repository.overheid.nl/sru"
    version: str = "2.0"
    operation: str = "searchRetrieve"
    max_records_per_request: int = 900
    repository_base_url: HttpUrl = "https://repository.officiele-overheidspublicaties.nl"
    zoek_base_url: HttpUrl = "https://zoek.officielebekendmakingen.nl"

class RateLimitSettings(BaseModel):
    """Rate limiting configuration."""
    request_delay: float = 2.0
    max_retries: int = 3  # Reduced from 5 - redirects won't resolve with retries
    delay_multiplier: float = 2.0
    successful_requests_to_reset: int = 5
    request_timeout: int = 30  # Increased to 30 seconds for slow government APIs
    connect_timeout: int = 10  # Separate connection timeout
    max_retry_delay: float = 10.0  # Maximum delay between retries

class FileSettings(BaseModel):
    """File handling configuration."""
    min_image_size_bytes: int = 50000
    min_pdf_size_bytes: int = 50000
    pdf_conversion_dpi: int = 300
    supported_extensions: List[str] = [".pdf", ".jpg", ".png", ".jpeg"]

class LoggingSettings(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file: str = "download.log"
    format: str = "%(asctime)s - %(levelname)s - %(message)s"

class Settings(BaseSettings):
    """Main settings class that combines all configuration groups."""
    api: APISettings = APISettings()
    date_range: DateRangeSettings = DateRangeSettings()
    directories: DirectorySettings
    sru: SRUSettings = SRUSettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    file: FileSettings = FileSettings()
    logging: LoggingSettings = LoggingSettings()
    # TODO: Add more keywords to exclude
    exclude_keywords: List[str] = [
        # "parkeerplaats", "laadpaal", "gehandicapt", "oplaadpunt",
        # "parkeerverbod", "parkeervergunning", "parkeerregime",
        # "parkeermogelijkheden", "parkeervoorzieningen",
        # "parkeersituatie", "parkeersituaties", "parkeerplaatsen",
        # "parkeerplaatsvoorzieningen"
    ]
    query_template: str = """(c.product-area==officielepublicaties AND 
        dt.modified>={date_start} AND dt.modified<={date_end} AND 
        dt.type = "verkeersbesluit " AND cql.allRecords =1 
        NOT dt.title any "{exclude_keywords}" AND 
        cql.allRecords=1 NOT dt.alternative any "{exclude_keywords}" )"""

    class Config:
        """Pydantic configuration."""
        env_prefix = "VERKEERSBESLUIT_"  # Environment variables should start with this
        env_nested_delimiter = "__"  # Use double underscore for nested settings

@lru_cache()
def get_settings() -> Settings:
    """
    Creates and caches a Settings instance.
    
    Returns:
        Settings instance with values from environment variables or defaults
    """
    return Settings(
        directories=DirectorySettings(
            verkeersbesluiten=Path(__file__).parent.parent.parent / "verkeersbesluiten",
            afbeeldingen=Path(__file__).parent.parent.parent / "afbeeldingen"
        )
    )