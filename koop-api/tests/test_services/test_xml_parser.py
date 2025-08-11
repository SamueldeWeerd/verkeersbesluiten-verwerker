import pytest
from src.utils.xml_parser import XMLParser
import xml.etree.ElementTree as ET

def test_extract_plain_text():
    parser = XMLParser()
    xml_string = """
        <root>
            <title>Test Title</title>
            <content>Test Content</content>
            <nested>
                <item>Nested Content</item>
            </nested>
        </root>
    """
    result = parser.extract_plain_text(xml_string)
    # Normalize whitespace for comparison
    expected = "Test Title Test Content Nested Content"
    actual = " ".join(result.split())
    assert expected == actual

def test_parse_metadata_block():
    parser = XMLParser()
    metadata_xml = ET.fromstring("""
        <root>
            <metadata name="simple.field" content="simple value"/>
            <metadata name="OVERHEIDop.gebiedsmarkering" content="Lijn">
                <metadata name="OVERHEIDop.geometrie" content="POINT(1 1)"/>
                <metadata name="OVERHEIDop.geometrieLabel" content="Test Label"/>
            </metadata>
        </root>
    """)
    
    result = parser.parse_metadata_block(metadata_xml)
    
    assert result["simple.field"] == "simple value"
    assert len(result["OVERHEIDop.gebiedsmarkering"]) == 1
    gebied = result["OVERHEIDop.gebiedsmarkering"][0]
    assert gebied["type"] == "Lijn"
    assert gebied["geometrie"] == "POINT(1 1)"
    assert gebied["label"] == "Test Label"

def test_extract_exb_code():
    parser = XMLParser()
    metadata = {
        "OVERHEIDop.externeBijlage": "some text exb-2024-12345 some more text"
    }
    
    result = parser.extract_exb_code(metadata)
    assert result == "exb-2024-12345"

def test_extract_exb_code_with_different_format():
    parser = XMLParser()
    metadata = {
        "OVERHEIDop.externeBijlage": "some text exb-24-12345 some more text"
    }
    
    result = parser.extract_exb_code(metadata)
    assert result == "exb-24-12345"

def test_extract_exb_code_not_found():
    parser = XMLParser()
    metadata = {
        "OVERHEIDop.externeBijlage": "some text without exb code"
    }
    
    result = parser.extract_exb_code(metadata)
    assert result is None

def test_extract_embedded_images():
    parser = XMLParser()
    xml_string = """
        <root>
            <illustratie naam="image1.jpg"/>
            <illustratie naam="image2.jpg"/>
            <illustratie/>
        </root>
    """
    
    result = parser.extract_embedded_images(xml_string)
    assert len(result) == 2
    assert "image1.jpg" in result
    assert "image2.jpg" in result