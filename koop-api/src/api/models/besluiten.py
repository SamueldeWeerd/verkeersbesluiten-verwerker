from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field

class GebiedsMarkeringModel(BaseModel):
    """Model for geographical area markings in a verkeersbesluit."""
    type: str = Field(..., description="Type of area marking (e.g., 'Lijn')")
    geometrie: Optional[str] = Field(None, description="Geometric data of the area")
    label: Optional[str] = Field(None, description="Label for the area marking")

class VerkeersBesluitMetadata(BaseModel):
    """Model for verkeersbesluit metadata."""
    bordcode: Optional[str] = Field(
        None,
        alias="OVERHEIDop.verkeersbordcode",
        description="Traffic sign code (bordcode) for filtering and classification"
    )
    gemeente: Optional[str] = Field(
        None,
        alias="OVERHEID.authority",
        description="Municipality/authority name (contains gemeente for municipal decisions)"
    )
    provincie: Optional[str] = Field(
        None,
        alias="DC.creator",
        description="Province/authority name (contains provincie for provincial decisions)"
    )
    gebiedsmarkering: Optional[List[GebiedsMarkeringModel]] = Field(
        None,
        alias="OVERHEIDop.gebiedsmarkering",
        description="List of geographical area markings"
    )
    externe_bijlage: Optional[str] = Field(
        None,
        alias="OVERHEIDop.externeBijlage",
        description="Reference to external attachment"
    )
    exb_code: Optional[str] = Field(
        None,
        description="Extracted external attachment code"
    )
    # Allow additional fields as metadata can contain various properties
    class Config:
        extra = "allow"

class VerkeersBesluitResponse(BaseModel):
    """Model for a single verkeersbesluit response."""
    id: str = Field(..., description="Unique identifier of the verkeersbesluit")
    text: str = Field(..., description="Full text content of the verkeersbesluit")
    metadata: VerkeersBesluitMetadata = Field(..., description="Metadata of the verkeersbesluit")
    images: List[HttpUrl] = Field(
        default_factory=list,
        description="List of URLs to associated images (maps, aerial photos)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "gmb-2024-12345",
                "text": "De burgemeester en wethouders van gemeente...",
                "metadata": {
                    "OVERHEIDop.verkeersbordcode": "C1",
                    "OVERHEID.authority": "Amsterdam",
                    "DC.creator": "Noord-Holland",
                    "OVERHEIDop.gebiedsmarkering": [
                        {
                            "type": "Lijn",
                            "geometrie": "POINT(4.8896 52.3740)",
                            "label": "Hoofdweg 123"
                        }
                    ],
                    "OVERHEIDop.externeBijlage": "exb-2024-67890",
                    "exb_code": "exb-2024-67890"
                },
                "images": [
                    "http://example.com/images/map1.png",
                    "http://example.com/images/aerial1.jpg"
                ]
            }
        }