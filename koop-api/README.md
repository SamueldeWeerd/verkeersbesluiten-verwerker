# KOOP API - Dutch Traffic Decisions Service

A FastAPI service that retrieves and filters Dutch government traffic decisions (verkeersbesluiten) from the KOOP API with intelligent CLIP-based image filtering.

## Quick Start

```bash
# Start with Docker Compose (recommended)
docker-compose up --build

# Access API
curl "http://localhost:8001/health"
```

**Service URL**: `http://localhost:8001` ([docs](http://localhost:8001/docs))

## API Endpoints

### GET `/besluiten/{start_date}/{end_date}`

Retrieves traffic decisions for a date range with optional filtering.

**Parameters:**
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `bordcode_categories` (optional): Filter by traffic sign categories (A, C, D, F, G)
- `provinces` (optional): Filter by Dutch provinces (case-insensitive)
- `gemeenten` (optional): Filter by municipalities (case-insensitive)

**Examples:**
```bash
# Basic date range
curl "http://localhost:8001/besluiten/2024-01-01/2024-01-02"

# With filters
curl "http://localhost:8001/besluiten/2024-01-01/2024-01-02?bordcode_categories=A&provinces=utrecht"
```

**Response:**
```json
[
  {
    "id": "gmb-2024-12345",
    "text": "De burgemeester en wethouders van gemeente...",
    "metadata": {
      "OVERHEIDop.verkeersbordcode": "C1",
      "OVERHEID.authority": "Amsterdam",
      "DC.creator": "Noord-Holland",
      "OVERHEIDop.gebiedsmarkering": [{
        "type": "Lijn",
        "geometrie": "POINT(4.8896 52.3740)",
        "label": "Hoofdweg 123"
      }],
      "OVERHEIDop.externeBijlage": "exb-2024-67890"
    },
    "images": [
      "http://localhost:8001/afbeeldingen/exb-2024-67890_page_1_bijlage.png"
    ]
  }
]
```

### GET `/health`
Health check endpoint.

## Intelligent Image Filtering

The service uses **OpenAI's CLIP model** to automatically classify and filter images:

### What Gets Downloaded
- ✅ **Maps**: Street maps, city maps, topographic maps, schematic maps
- ✅ **Aerial/Satellite Images**: Aerial photographs, satellite imagery, bird's eye views
- ❌ **Text/Signs**: Text documents, road signs, traffic signs, forms

### Benefits
- **Storage Efficiency**: Only relevant images saved
- **Bandwidth Optimization**: Prevents unnecessary downloads
- **Data Quality**: Improves downstream processing

### Classification Process
1. PDF attachments converted to images
2. Each image classified using CLIP model
3. Only maps/aerial images saved to `afbeeldingen/` directory
4. Classification statistics logged

**Example Log Output:**
```
✅ Afbeelding opgeslagen (kaart/luchtfoto): afbeeldingen/image123.png
⏩ Overgeslagen pagina 2 (geen kaart/luchtfoto)
📊 1/2 pagina's opgeslagen voor exb-2024-123
```

## Configuration

### Environment Variables
```bash
# API Settings
VERKEERSBESLUIT_API__HOST=0.0.0.0
VERKEERSBESLUIT_API__PORT=8000

# Rate Limiting
VERKEERSBESLUIT_RATE_LIMIT__REQUEST_TIMEOUT=30
VERKEERSBESLUIT_RATE_LIMIT__MAX_RETRIES=3

# Logging
VERKEERSBESLUIT_LOGGING__LEVEL=INFO
```

### CLIP Model Configuration
```python
# Confidence threshold (adjustable in code)
confidence_threshold = 0.4  # Lower = more permissive

# Classification prompts
classification_prompts = [
    "a map, a schematic map, a city map, a road map, a topographic map",
    "a satellite image, an aerial photograph, an aerial view, a bird's eye view", 
    "text document, a road sign, a traffic sign, plain text, a form"
]
```

## Features

### Rate Limiting & Resilience
- Adaptive rate limiting with exponential backoff
- Automatic retries for failed requests
- Configurable timeouts and retry limits

### Filtering Capabilities
- **Bordcode Categories**: A, C, D, F, G (case insensitive)
- **Provinces**: All Dutch provinces (case insensitive)
- **Municipalities**: Any gemeente name (case insensitive)
- Early filtering before image processing

### Image Processing
- Automatic PDF to image conversion
- CLIP model classification (maps/aerial vs text/signs)
- Local storage in `afbeeldingen/` directory
- Detailed processing logs

## Technical Details

### Dependencies
```bash
# Core API
fastapi
uvicorn
httpx

# Image Processing
Pillow
pdf2image

# AI Classification
torch
torchvision
clip-by-openai

# Data Processing
lxml
shapely
```

### Docker Network
Runs on `n8n-network`, accessible as `koop-api-service:8001` to other containers.

### Performance
- **First run**: Downloads CLIP model (~400MB)
- **Image classification**: ~0.1-0.5 seconds per image
- **GPU support**: Optional but recommended for faster classification
- **Memory usage**: ~2-4GB (with CLIP model loaded)

## Development

### Local Setup
```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001
```

### Project Structure
```
src/
├── api/
│   ├── main.py              # FastAPI application
│   └── routes/              # API endpoints
├── services/
│   └── besluit_download_service.py  # Core logic
├── utils/
│   ├── filters.py           # Filter implementations
│   ├── http_client.py       # Rate-limited HTTP client
│   └── xml_parser.py        # XML processing
├── ml/
│   └── clip_classifier.py   # CLIP image classification
└── config/
    └── settings.py          # Configuration
```

### Testing CLIP Classification
```bash
# Test with specific image
python test_image_classifier.py path/to/image.png

# View classification results
{
  'is_map_or_aerial': True,
  'confidence': 0.87,
  'classification': 'maps',
  'probabilities': {
    'maps': 0.87,
    'aerial_satellite': 0.08,
    'text_signs': 0.05
  }
}
```

## Troubleshooting

**CLIP Model Issues:**
```bash
# Install missing dependencies
pip install ftfy regex tqdm clip-by-openai

# Force CPU usage if GPU memory issues
# Set device="cpu" in image_classifier.py
```

**Rate Limiting:**
- Service implements exponential backoff
- Check logs for retry information
- Adjust timeouts in environment variables

**Image Processing:**
- Ensure sufficient disk space for images
- Check `afbeeldingen/` directory permissions
- Monitor classification statistics in logs

## Integration Notes

This service is designed to work with the broader traffic decree processing pipeline:
1. **Input**: Date ranges and filter criteria
2. **Processing**: Fetches decisions, converts PDFs, classifies images
3. **Output**: Structured data with filtered, relevant images
4. **Downstream**: Feature matching API processes the saved images

The CLIP classification prevents downloading irrelevant images (text, signs) that would otherwise consume storage and processing resources in later pipeline stages.