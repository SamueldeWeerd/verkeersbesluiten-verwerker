import pytest
from pydantic import ValidationError
from src.api.models.besluiten import (
    GebiedsMarkeringModel,
    VerkeersBesluitMetadata,
    VerkeersBesluitResponse
)

def test_gebied_marking_model():
    data = {
        "type": "Lijn",
        "geometrie": "POINT(4.8896 52.3740)",
        "label": "Hoofdweg 123"
    }
    model = GebiedsMarkeringModel(**data)
    assert model.type == "Lijn"
    assert model.geometrie == "POINT(4.8896 52.3740)"
    assert model.label == "Hoofdweg 123"

def test_metadata_model():
    data = {
        "OVERHEIDop.bordcode": "C1",
        "OVERHEIDop.gemeente": "Amsterdam",
        "OVERHEIDop.provincie": "Noord-Holland",
        "OVERHEIDop.gebiedsmarkering": [
            {
                "type": "Lijn",
                "geometrie": "POINT(4.8896 52.3740)",
                "label": "Hoofdweg 123"
            }
        ],
        "OVERHEIDop.externeBijlage": "exb-2024-67890",
        "exb_code": "exb-2024-67890",
        "extra_field": "allowed"  # Should be allowed due to extra = "allow"
    }
    model = VerkeersBesluitMetadata(**data)
    assert model.bordcode == "C1"
    assert model.gemeente == "Amsterdam"
    assert model.provincie == "Noord-Holland"
    assert len(model.gebiedsmarkering) == 1
    assert model.externe_bijlage == "exb-2024-67890"
    assert model.exb_code == "exb-2024-67890"
    assert model.extra_field == "allowed"

def test_verkeersbesluit_response():
    data = {
        "id": "gmb-2024-12345",
        "text": "Test content",
        "metadata": {
            "OVERHEIDop.bordcode": "C1",
            "OVERHEIDop.gemeente": "Amsterdam",
            "OVERHEIDop.provincie": "Noord-Holland",
            "OVERHEIDop.externeBijlage": "exb-2024-67890",
            "exb_code": "exb-2024-67890"
        },
        "images": [
            "http://example.com/image1.png",
            "http://example.com/image2.jpg"
        ]
    }
    model = VerkeersBesluitResponse(**data)
    assert model.id == "gmb-2024-12345"
    assert model.text == "Test content"
    assert model.metadata.bordcode == "C1"
    assert model.metadata.gemeente == "Amsterdam"
    assert model.metadata.provincie == "Noord-Holland"
    assert len(model.images) == 2

def test_invalid_image_url():
    data = {
        "id": "gmb-2024-12345",
        "text": "Test content",
        "metadata": {},
        "images": ["not-a-url"]  # Should fail validation
    }
    with pytest.raises(ValidationError):
        VerkeersBesluitResponse(**data)