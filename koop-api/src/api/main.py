from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging

from src.api.routes import download_besluiten, health, maintenance
from src.config.settings import get_settings

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format=settings.logging.format,
    handlers=[
        logging.StreamHandler(),  # Console output for Docker logs
        logging.FileHandler(settings.logging.file)  # File output
    ],
    force=True  # Override any existing logging configuration
)

# Ensure root logger level is set
logging.getLogger().setLevel(getattr(logging, settings.logging.level))

# Test logging configuration
logging.info("🚀 Logging configured successfully - detailed processing logs will be visible")

app = FastAPI(
    title="Verkeersbesluiten API",
    description="API for retrieving and processing traffic decisions (verkeersbesluiten).",
    version="1.1.0"
)

# Mount static files for serving saved images
app.mount("/afbeeldingen", StaticFiles(directory="afbeeldingen"), name="afbeeldingen")

# Mount routes
app.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

app.include_router(
    download_besluiten.router,
    prefix="/besluiten",
    tags=["besluiten"]
)

app.include_router(
    maintenance.router,
    prefix="/maintenance",
    tags=["maintenance"]
)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False  # Disable reload in container to avoid constant reloading
    )