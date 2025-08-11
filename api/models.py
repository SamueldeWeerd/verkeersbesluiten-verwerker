"""
Data models for the traffic decree processing API
"""
from enum import Enum
from pydantic import BaseModel, Field
from datetime import date
from typing import List, Optional


class BordcodeCategory(str, Enum):
    """Valid bordcode categories for filtering traffic decrees (case-insensitive)"""
    A = "A"
    C = "C"
    D = "D"
    F = "F"
    G = "G"

    @classmethod
    def _missing_(cls, value):
        """Allow case-insensitive lookup"""
        if isinstance(value, str):
            upper_value = value.upper()
            for member in cls:
                if member.value == upper_value:
                    return member
        return None


class Province(str, Enum):
    """Valid Dutch provinces for filtering traffic decrees"""
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


class WorkflowPayload(BaseModel):
    """Payload for triggering the N8N traffic decree workflow"""
    
    start_date: date = Field(
        ..., 
        description="The start date from which traffic decrees should be processed.", 
        example="2024-05-21"
    )
    end_date: date = Field(
        ..., 
        description="The end date until which traffic decrees should be processed.", 
        example="2024-05-22"
    )
    bordcode_categories: Optional[List[BordcodeCategory]] = Field(
        None, 
        description="Filter by bordcode categories (A, C, D, F, G). Include if metadata contains ANY of these letters.",
        example=["A", "C"]
    )
    provinces: Optional[List[Province]] = Field(
        None, 
        description="Filter by Dutch provinces (case-insensitive).",
        example=["noord-holland", "utrecht"]
    )
    gemeenten: Optional[List[str]] = Field(
        None, 
        description="Filter by municipalities (case-insensitive). Include decisions from these specific municipalities.",
        example=["amsterdam", "utrecht"]
    )