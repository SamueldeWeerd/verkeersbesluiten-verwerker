"""
Filtering utilities for verkeersbesluit data.

This module contains utility functions for filtering traffic decisions based on
various criteria like bordcode categories, provinces, and municipalities.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
import logging


class BordcodeCategory(str, Enum):
    """Valid bordcode categories for filtering."""
    A = "A"
    C = "C" 
    D = "D"
    F = "F"
    G = "G"


class Province(str, Enum):
    """Dutch provinces for filtering."""
    DRENTHE = "drenthe"
    FLEVOLAND = "flevoland"
    FRIESLAND = "friesland"
    GELDERLAND = "gelderland"
    GRONINGEN = "groningen"
    LIMBURG = "limburg"
    NOORD_BRABANT = "noord-brabant"
    NOORD_HOLLAND = "noord-holland"
    OVERIJSSEL = "overijssel"
    UTRECHT = "utrecht"
    ZEELAND = "zeeland"
    ZUID_HOLLAND = "zuid-holland"


def validate_provinces(provinces: List[str]) -> List[str]:
    """
    Validate province names against known Dutch provinces.
    
    Args:
        provinces: List of province names to validate
        
    Returns:
        List of invalid province names (empty if all valid)
        
    Raises:
        ValueError: If any provinces are invalid
    """
    if not provinces:
        return []
    
    valid_provinces = [p.value for p in Province]
    invalid_provinces = [p for p in provinces if p.lower() not in valid_provinces]
    
    if invalid_provinces:
        raise ValueError(
            f"Invalid provinces: {', '.join(invalid_provinces)}. "
            f"Valid options: {', '.join(valid_provinces)}"
        )
    
    return []


def check_bordcode_filter(
    metadata: Dict[str, Any], 
    bordcode_categories: Optional[List[BordcodeCategory]], 
    besluit_id: str
) -> bool:
    """
    Check if a besluit passes the bordcode categories filter.
    
    Args:
        metadata: Besluit metadata dictionary
        bordcode_categories: List of required bordcode categories (A, C, D, F, G)
        besluit_id: ID of the besluit for logging
        
    Returns:
        True if besluit passes filter (or no filter applied), False otherwise
    """
    if not bordcode_categories:
        return True
    
    # Check multiple possible bordcode fields
    bordcode_value = (
        metadata.get("OVERHEIDop.verkeersbordcode", "") 
    )
    contains_category = any(
        category.value in bordcode_value.upper() 
        for category in bordcode_categories
    )
    
    if not contains_category:
        logging.info(
            f"🚫 {besluit_id}: Excluded by bordcode filter "
            f"(has '{bordcode_value}', need any of: {[c.value for c in bordcode_categories]})"
        )
        return False
    
    return True


def check_province_filter(
    metadata: Dict[str, Any], 
    provinces: Optional[List[str]], 
    besluit_id: str
) -> bool:
    """
    Check if a besluit passes the province filter.
    
    Checks multiple fields for province information including OVERHEID.authority,
    DC.creator (case-insensitive).
    
    Args:
        metadata: Besluit metadata dictionary
        provinces: List of required provinces
        besluit_id: ID of the besluit for logging
        
    Returns:
        True if besluit passes filter (or no filter applied), False otherwise
    """
    if not provinces:
        return True
    
    # Check multiple possible province/authority fields
    authority_value = metadata.get("OVERHEID.authority", "").lower()
    creator_value = metadata.get("DC.creator", "").lower()
    
    # Check if ANY of the specified provinces match in ANY of the fields
    province_match = any(
        prov.lower() in authority_value or 
        prov.lower() in creator_value 
        for prov in provinces
    )
    
    if not province_match:
        logging.info(
            f"🚫 {besluit_id}: Excluded by province filter "
            f"(has authority: '{authority_value}', creator: '{creator_value}', "
            f"need any of: {[p.lower() for p in provinces]})"
        )
        return False
    
    return True


def check_gemeente_filter(
    metadata: Dict[str, Any], 
    gemeenten: Optional[List[str]], 
    besluit_id: str
) -> bool:
    """
    Check if a besluit passes the gemeente (municipality) filter.
    
    This function performs case-insensitive matching against multiple fields
    that might contain municipality information. For municipal decisions,
    the authority/creator fields often contain the municipality name.
    
    Args:
        metadata: Besluit metadata dictionary
        gemeenten: List of required municipalities
        besluit_id: ID of the besluit for logging
        
    Returns:
        True if besluit passes filter (or no filter applied), False otherwise
    """
    if not gemeenten:
        return True
    
    # Check multiple possible gemeente/authority fields
    authority_value = metadata.get("OVERHEID.authority", "").lower()
    creator_value = metadata.get("DC.creator", "").lower()
    org_type = metadata.get("OVERHEID.organisationType", "").lower()
    
    # Check if ANY of the specified gemeenten match (case-insensitive partial match)
    gemeente_match = any(
        gem.lower() in authority_value or
        gem.lower() in creator_value
        for gem in gemeenten
    )
    
    if not gemeente_match:
        logging.info(
            f"🚫 {besluit_id}: Excluded by gemeente filter "
            f"(has authority: '{authority_value}', creator: '{creator_value}', "
            f"need any of: {[g.lower() for g in gemeenten]})"
        )
        return False
    
    return True


def apply_filters(
    besluiten: List[Dict[str, Any]],
    bordcode_categories: Optional[List[BordcodeCategory]] = None,
    provinces: Optional[List[str]] = None,
    gemeenten: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Apply multiple filters to a list of verkeersbesluit data.
    
    Args:
        besluiten: List of besluit dictionaries to filter
        bordcode_categories: Optional bordcode categories filter
        provinces: Optional provinces filter
        gemeenten: Optional municipalities filter
        
    Returns:
        Filtered list of besluiten
        
    Raises:
        ValueError: If invalid provinces are provided
    """
    if not any([bordcode_categories, provinces, gemeenten]):
        logging.info(f"📄 No filters applied - returning all {len(besluiten)} decisions")
        return besluiten
    
    # Validate provinces if provided
    if provinces:
        validate_provinces(provinces)
    
    logging.info(
        f"🔍 Applying filters - Bordcode categories: "
        f"{[c.value for c in bordcode_categories] if bordcode_categories else None}, "
        f"Provinces: {provinces}, Gemeenten: {gemeenten}"
    )
    
    filtered_results = []
    total_before_filtering = len(besluiten)
    excluded_bordcode_count = 0
    excluded_province_count = 0
    excluded_gemeente_count = 0
    
    for besluit in besluiten:
        metadata = besluit.get("metadata", {})
        besluit_id = besluit.get("id", "unknown")
        
        # Apply all filters - besluit must pass ALL active filters
        if not check_bordcode_filter(metadata, bordcode_categories, besluit_id):
            excluded_bordcode_count += 1
            continue
            
        if not check_province_filter(metadata, provinces, besluit_id):
            excluded_province_count += 1
            continue
            
        if not check_gemeente_filter(metadata, gemeenten, besluit_id):
            excluded_gemeente_count += 1
            continue
        
        # If we get here, the besluit passed all filters
        logging.info(f"✅ {besluit_id}: Included (passed all filters)")
        filtered_results.append(besluit)
    
    # Summary logging
    logging.info(f"📊 Filter summary: {len(filtered_results)}/{total_before_filtering} decisions included")
    if excluded_bordcode_count > 0:
        logging.info(f"🚫 Excluded {excluded_bordcode_count} decisions due to bordcode filter")
    if excluded_province_count > 0:
        logging.info(f"🚫 Excluded {excluded_province_count} decisions due to province filter")
    if excluded_gemeente_count > 0:
        logging.info(f"🚫 Excluded {excluded_gemeente_count} decisions due to gemeente filter")
    
    return filtered_results