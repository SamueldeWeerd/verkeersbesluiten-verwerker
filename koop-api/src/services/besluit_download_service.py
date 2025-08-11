from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import xml.etree.ElementTree as ET
import os
from io import BytesIO
from pdf2image import convert_from_bytes

from src.config.settings import Settings, get_settings
from src.utils.http_client import RateLimitedClient
from src.utils.xml_parser import XMLParser
from src.ml.clip_classifier import ImageClassifier
from src.utils.filters import BordcodeCategory, check_bordcode_filter, check_province_filter, check_gemeente_filter, validate_provinces

class BesluitService:
    """
    Service for handling verkeersbesluit operations.
    Coordinates between HTTP client, XML parser, and image classification.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        http_client: Optional[RateLimitedClient] = None,
        xml_parser: Optional[XMLParser] = None,
        image_classifier: Optional[ImageClassifier] = None
    ):
        """
        Initialize the service with its dependencies.
        If not provided, will create instances using settings.
        """
        self._settings = settings or get_settings()
        self._http_client = http_client or RateLimitedClient(settings=self._settings)
        self._xml_parser = xml_parser or XMLParser()
        self._image_classifier = image_classifier or ImageClassifier(settings=self._settings)
    
    def get_besluiten_for_date(
        self, 
        start_date_str: str, 
        end_date_str: str,
        bordcode_categories: Optional[List[BordcodeCategory]] = None,
        provinces: Optional[List[str]] = None,
        gemeenten: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetches and processes verkeersbesluiten for a specific date with optional filtering.
        Filtering is applied BEFORE image processing for efficiency.
        
        Args:
            start_date_str: Start date in YYYY-MM-DD format
            end_date_str: End date in YYYY-MM-DD format
            bordcode_categories: Optional bordcode categories filter
            provinces: Optional provinces filter
            gemeenten: Optional municipalities filter
            
        Returns:
            List of processed verkeersbesluit data (already filtered)
        """
        # Validate date format
        try:
            datetime.strptime(start_date_str, "%Y-%m-%d")
            datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format (YYYY-MM-DD)")
        
        # Prepare SRU query
        query = self._settings.query_template.format(
            date_start=start_date_str,
            date_end=end_date_str,
            exclude_keywords=" ".join(self._settings.exclude_keywords)
        )
        
        # Make SRU request
        params = {
            "version": self._settings.sru.version,
            "operation": self._settings.sru.operation,
            "query": query,
            "maximumRecords": str(self._settings.sru.max_records_per_request)
        }
        
        response = self._http_client.get(str(self._settings.sru.base_url), params=params)
        if not response or not response.ok:
            logging.warning(f"⚠️ Failed to get SRU data for {start_date_str} to {end_date_str}")
            return []
        
        # Parse response and extract records
        records = self._xml_parser.parse_sru_response(response.content)
        if not records:
            return []
        
        # Process each record
        all_besluiten = []
        total_records = len(records)
        logging.info(f"📄 Processing {total_records} verkeersbesluit records...")
        
        # Log filter configuration
        if bordcode_categories or provinces or gemeenten:
            logging.info(
                f"🔍 Applying filters - Bordcode categories: "
                f"{[c.value for c in bordcode_categories] if bordcode_categories else None}, "
                f"Provinces: {provinces}, Gemeenten: {gemeenten}"
            )
        else:
            logging.info("📄 No filters applied - processing all records")
        
        for i, record in enumerate(records, 1):
            urls = self._xml_parser.extract_urls_from_record(record)
            if not urls.get("content"):
                logging.warning(f"⚠️ Record {i}/{total_records}: No content URL found, skipping...")
                continue
            
            # Get and process content
            content_url = urls["content"]
            besluit_id = content_url.split("/")[-1].replace(".xml", "")
            logging.info(f"📖 Processing {i}/{total_records}: {besluit_id}")
            
            content_response = self._http_client.get(content_url)
            if not content_response or not content_response.ok:
                logging.warning(f"❌ Failed to download content for {besluit_id}")
                continue
            
            content = content_response.content.decode("utf-8", errors="ignore")
            
            # Check exclusion keywords
            excluded_keywords = [k for k in self._settings.exclude_keywords if k in content.lower()]
            if excluded_keywords:
                logging.info(f"🚫 {besluit_id}: Excluded (contains: {', '.join(excluded_keywords)})")
                continue
            
            # Get metadata if available
            metadata = {}
            if metadata_url := urls.get("metadata"):
                meta_response = self._http_client.get(metadata_url)
                if meta_response and meta_response.ok:
                    metadata = self._xml_parser.parse_metadata_block(
                        ET.fromstring(meta_response.content)
                    )
            
            # Apply filters BEFORE expensive image processing
            if bordcode_categories or provinces or gemeenten:
                # Validate provinces if provided
                if provinces:
                    validate_provinces(provinces)
                
                # Check each filter - if any fails, skip this besluit
                if not check_bordcode_filter(metadata, bordcode_categories, besluit_id):
                    continue
                    
                if not check_province_filter(metadata, provinces, besluit_id):
                    continue
                    
                if not check_gemeente_filter(metadata, gemeenten, besluit_id):
                    continue
                
                # If we get here, the besluit passed all filters
                logging.info(f"✅ {besluit_id}: Passed filters - proceeding with image processing")
            
            # Extract images (only for filtered besluiten)
            image_urls = []
            logging.info(f"🔍 {besluit_id}: Scanning for images...")
            
            # Handle PDF attachments
            if exb_code := self._xml_parser.extract_exb_code(metadata):
                pdf_url = f"{self._settings.sru.repository_base_url}/externebijlagen/{exb_code}/1/bijlage/{exb_code}.pdf"
                logging.info(f"📎 {besluit_id}: Found PDF attachment with exb_code: {exb_code}")
                
                # Download and check if it's a map/aerial photo
                saved_image_url = self._download_and_save_pdf_attachment(pdf_url, exb_code, besluit_id)
                if saved_image_url:
                    image_urls.append(saved_image_url)
                    logging.info(f"✅ {besluit_id}: PDF contains map/aerial photo - saved locally")
                else:
                    logging.info(f"⏩ {besluit_id}: PDF does not contain map/aerial photo - skipped")
            
            # Handle embedded images
            embedded_images = self._xml_parser.extract_embedded_images(content)
            if embedded_images:
                logging.info(f"🖼️ {besluit_id}: Found {len(embedded_images)} embedded image(s)")
                for image_name in embedded_images:
                    image_url = f"{self._settings.sru.zoek_base_url}/{image_name}"
                    image_urls.append(image_url)
                    logging.info(f"   📷 Added embedded image: {image_name}")
            else:
                logging.info(f"📷 {besluit_id}: No embedded images found")
            
            # Combine all data
            besluit_data = {
                "id": besluit_id,
                "text": self._xml_parser.extract_plain_text(content),
                "metadata": metadata,
                "images": image_urls
            }
            all_besluiten.append(besluit_data)
            
            # Summary logging for each processed besluit
            image_count = len(image_urls)
            if image_count > 0:
                logging.info(f"✅ {besluit_id}: Completed processing with {image_count} image(s)")
            else:
                logging.info(f"📝 {besluit_id}: Completed processing (no images)")
        
        logging.info(f"🏁 Finished processing {len(all_besluiten)}/{total_records} verkeersbesluit records")
        return all_besluiten
    
    def _download_and_save_pdf_attachment(self, pdf_url: str, exb_code: str, besluit_id: str) -> str:
        """
        Downloads a PDF attachment and converts its first page to an image.
        Uses the PDF's exb_code for downloading but saves with the verkeersbesluit's ID.
        Returns the API URL to access the saved image, or empty string if no image was saved.
        """
        
        logging.info(f"⬇️ Downloading and converting: {pdf_url}")
        
        try:
            pdf_response = self._http_client.get(pdf_url)
            if not pdf_response or not pdf_response.ok:
                logging.warning(f"❌ Failed to download PDF from {pdf_url}")
                return ""
            
            if len(pdf_response.content) < self._settings.file.min_pdf_size_bytes:
                logging.warning(f"❌ PDF too small ({len(pdf_response.content)} bytes)")
                return ""
            
            # Convert PDF bytes directly to images
            images = convert_from_bytes(pdf_response.content, dpi=self._settings.file.pdf_conversion_dpi)
            
            if not images:
                logging.warning(f"❌ No pages found in PDF for {exb_code}")
                return ""
            
            # Only process the first page
            first_page = images[0]
            
            # Check if it's a map/aerial photo using CLIP
            if not self._image_classifier.should_download_image(first_page):
                logging.info(f"⏩ PDF does not contain map/aerial photo")
                return ""
            
            # Save the image locally
            try:
                # Ensure afbeeldingen directory exists
                afbeeldingen_dir = self._settings.directories.afbeeldingen
                os.makedirs(afbeeldingen_dir, exist_ok=True)
                
                # Use the verkeersbesluit ID for the filename, not the PDF's exb_code
                output_filename = f"{besluit_id}_page_1_bijlage.png"
                output_path = os.path.join(afbeeldingen_dir, output_filename)
                
                first_page.save(output_path, "PNG")
                
                # Return the external API-accessible URL (for Docker network access)
                relative_path = f"afbeeldingen/{output_filename}"
                image_url = f"{self._settings.api.external_base_url}/{relative_path}"
                
                logging.info(f"✅ Saved first page (map/aerial photo): {image_url}")
                return image_url
                
            except Exception as e:
                logging.warning(f"⚠️ Error saving image: {e}")
                return ""
                
        except Exception as e:
            logging.warning(f"⚠️ Error processing PDF: {e}")
            return ""