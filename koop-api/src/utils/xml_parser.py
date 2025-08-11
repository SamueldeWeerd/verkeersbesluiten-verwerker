import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
import logging
import re

class XMLParser:
    """
    Handles all XML parsing operations for verkeersbesluit data.
    Encapsulates XML parsing logic and provides clean interfaces for extracting data.
    """
    
    def __init__(self):
        self.ns = {
            "sru": "http://docs.oasis-open.org/ns/search-ws/sruResponse",
            "gzd": "http://standaarden.overheid.nl/sru"
        }
    
    def extract_plain_text(self, xml_string: str) -> str:
        """
        Extracts plain text content from an XML string.
        
        Args:
            xml_string: Raw XML content
            
        Returns:
            Extracted text with whitespace normalized
        """
        try:
            root = ET.fromstring(xml_string)
            return " ".join(root.itertext()).strip()
        except ET.ParseError as e:
            logging.error(f"❌ XML Parse error: {e}")
            return ""
    
    def parse_metadata_block(self, meta_root: ET.Element) -> Dict[str, Any]:
        """
        Parses a metadata block from XML into a structured dictionary.
        
        Args:
            meta_root: XML Element containing metadata
            
        Returns:
            Dictionary containing parsed metadata
        """
        metadata = {}
        gebiedsmarkeringen = []
        
        for m in meta_root.findall("metadata"):
            name = m.attrib.get("name")
            content = m.attrib.get("content")
            
            if name == "OVERHEIDop.gebiedsmarkering":
                gebied = {"type": content}
                for sub in m.findall("metadata"):
                    sub_name = sub.attrib.get("name")
                    sub_content = sub.attrib.get("content")
                    if sub_name == "OVERHEIDop.geometrie":
                        gebied["geometrie"] = sub_content
                    elif sub_name == "OVERHEIDop.geometrieLabel":
                        gebied["label"] = sub_content
                gebiedsmarkeringen.append(gebied)
            elif name and content:
                metadata[name] = content.strip()
        
        if gebiedsmarkeringen:
            metadata["OVERHEIDop.gebiedsmarkering"] = gebiedsmarkeringen
        
        return metadata
    
    def extract_urls_from_record(self, record: ET.Element) -> Dict[str, str]:
        """
        Extracts content and metadata URLs from a record.
        
        Args:
            record: XML Element containing record data
            
        Returns:
            Dictionary with 'content' and 'metadata' URLs
        """
        urls = {}
        for item_url in record.findall(f".//gzd:itemUrl", self.ns):
            manifestation = item_url.attrib.get("manifestation")
            if manifestation == "xml":
                urls["content"] = item_url.text
            elif manifestation == "metadata":
                urls["metadata"] = item_url.text
        return urls
    
    def extract_embedded_images(self, xml_string: str) -> List[str]:
        """
        Extracts embedded image references from XML content.
        
        Args:
            xml_string: Raw XML content
            
        Returns:
            List of image identifiers/names
        """
        try:
            root = ET.fromstring(xml_string)
            images = []
            for ill in root.findall(".//illustratie"):
                naam = ill.attrib.get("naam")
                if naam:
                    images.append(naam)
            return images
        except ET.ParseError as e:
            logging.error(f"❌ XML parsing error: {e}")
            return []
    
    def parse_sru_response(self, xml_content: bytes) -> List[ET.Element]:
        """
        Parses an SRU response and extracts all records.
        
        Args:
            xml_content: Raw XML content in bytes
            
        Returns:
            List of record Elements
        """
        try:
            root = ET.fromstring(xml_content)
            return root.findall(".//sru:recordData", self.ns)
        except ET.ParseError as e:
            logging.error(f"❌ XML parsing error: {e}")
            return []
    
    def extract_exb_code(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extracts the externe bijlage code from metadata.
        
        Args:
            metadata: Parsed metadata dictionary
            
        Returns:
            exb code if found, None otherwise
        """
        if externe_bijlage := metadata.get("OVERHEIDop.externeBijlage"):
            if match := re.search(r"exb-\d+-\d+", externe_bijlage):
                return match.group(0)
        return None