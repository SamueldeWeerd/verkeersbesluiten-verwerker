from fastapi import APIRouter, HTTPException
import os
import shutil
import logging
from pathlib import Path

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/remove-images")
async def remove_images():
    """Clean the afbeeldingen directory by removing all files."""
    try:
        image_dir = Path("afbeeldingen")
        if not image_dir.exists():
            return {"status": "success", "message": "Directory does not exist, nothing to clean"}

        # Count files before cleaning
        files = list(image_dir.glob("*"))
        file_count = len(files)
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        size_mb = total_size / (1024 * 1024)  # Convert to MB

        # Remove all files
        for file in files:
            if file.is_file():
                file.unlink()
                logger.info(f"Deleted file: {file.name}")

        return {
            "status": "success",
            "message": f"Successfully cleaned {file_count} files ({size_mb:.2f} MB) from afbeeldingen directory"
        }

    except Exception as e:
        logger.error(f"Error cleaning afbeeldingen directory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clean directory: {str(e)}"
        )