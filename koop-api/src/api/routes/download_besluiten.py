from fastapi import APIRouter, HTTPException, Path, Query
from datetime import datetime
from typing import List, Optional
import logging

from src.services.besluit_download_service import BesluitService
from src.config.settings import get_settings
from src.api.models.besluiten import VerkeersBesluitResponse
from src.utils.filters import BordcodeCategory

router = APIRouter()
settings = get_settings()
besluit_service = BesluitService(settings=settings)

@router.get("/{start_date_str}/{end_date_str}", summary="Get traffic decisions for a specific date range")
async def get_besluiten_by_date(
    start_date_str: str = Path(..., description="Date in YYYY-MM-DD format", regex=r"^\d{4}-\d{2}-\d{2}$"),
    end_date_str: str = Path(..., description="Date in YYYY-MM-DD format", regex=r"^\d{4}-\d{2}-\d{2}$"),
    bordcode_categories: Optional[List[BordcodeCategory]] = Query(None, description="Filter by bordcode categories (A, C, D, F, G). Include if metadata contains ANY of these letters."),
    provinces: Optional[List[str]] = Query(None, description="Filter by Dutch provinces (case-insensitive). Valid values: drenthe, flevoland, friesland, gelderland, groningen, limburg, noord-brabant, noord-holland, overijssel, utrecht, zeeland, zuid-holland"),
    gemeenten: Optional[List[str]] = Query(None, description="Filter by municipalities (case-insensitive). Include decisions from these specific municipalities."), 
    exclude_keywords: Optional[List[str]] = Query(None, description="Exclude keywords from the search. Separate with commas.")
) -> List[VerkeersBesluitResponse]:
    """
    Retrieves all traffic decisions for a given date range with optional filtering.
    
    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format  
        bordcode_categories: Optional list of bordcode categories (A, B, C, D, E, F, G, H, J, K, L). 
                           Includes decisions if metadata contains ANY of these letters.
        provinces: Optional list of Dutch provinces (case-insensitive)
        gemeenten: Optional list of municipalities (case-insensitive)
        
    Returns:
        List of processed verkeersbesluit data including metadata, text, and image URLs
        
    Important: 
        If you set the name of a province, it will only return besluiten from the province, not the municipalities in that province.
        If you set the name of a municipality, it will only return besluiten from the municipality, not the province.
    """
    try:
        # Pass filters directly to service for early filtering (before image processing)
        results = besluit_service.get_besluiten_for_date(
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            bordcode_categories=bordcode_categories,
            provinces=provinces,
            gemeenten=gemeenten,
            exclude_keywords=exclude_keywords
        )
        
        return results
    except HTTPException:
        # Re-raise HTTPExceptions (like our validation errors) without modification
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")