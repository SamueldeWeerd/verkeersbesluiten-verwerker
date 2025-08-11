# Feature Matching API - AI-Powered Map Processing Service

Advanced map processing service combining feature matching, map cutting, and georeferencing using KAZE features and GIM+Roma deep learning models.

## Quick Start

```bash
# Start service
docker-compose up -d

# Health check
curl http://localhost:8005/health
```

**Service URL**: `http://localhost:8005` ([docs](http://localhost:8005/docs))

## Core Features

### 🔍 **Dual AI Matching Engines**
- **KAZE Features**: Optimized for schematic maps and technical diagrams
- **GIM+Roma Model**: Deep learning for aerial imagery (luchtfoto) matching
- **Automatic Selection**: Chooses best algorithm based on map type
- **Quality Assessment**: Inlier ratio analysis and confidence scoring

### 🗺️ **Multi-Source Map Cutting**
- **Map Sources**: OpenStreetMap, Dutch BGT, BRT-A, Luchtfoto, BAG buildings
- **Geometric Buffering**: True buffer distances around input geometries
- **Format Support**: GeoJSON, WKT, coordinate lists
- **Auto Georeferencing**: World file (.pgw) and GeoTIFF generation

### ⚡ **Performance Optimizations**
- **Memory Management**: Efficient processing with automatic cleanup
- **Model Caching**: Persistent checkpoints (avoids ~1.2GB re-download)
- **CPU/GPU Support**: Automatic device detection
- **Buffer Optimization**: Tests multiple buffer sizes for best matches

## API Endpoints

### POST `/cutout-and-match`
Combined workflow: cuts map sections and performs feature matching with buffer optimization.

**Parameters:**
- `source_image` (file): Image to match
- `geometry` (string): GeoJSON/WKT/coordinates
- `map_type` (string): Map source (osm, bgt-omtrek, luchtfoto, etc.)
- `overlay_transparency` (float): 0.0-1.0
- `traffic_decree_id` (optional): Session identifier

**Example:**
```bash
curl -X POST "http://localhost:8005/cutout-and-match" \
  -F "source_image=@source.png" \
  -F "geometry=POINT(136000 455000)" \
  -F "map_type=osm" \
  -F "overlay_transparency=0.7"
```

### POST `/cutout-and-match-with-url`
Same as above but accepts image URL instead of file upload.

### POST `/match-maps`
Feature matching between two images with optional georeferencing.

**Parameters:**
- `source_image` (file): Source image
- `destination_image` (file): Reference image  
- `destination_pgw` (file): World file for georeferencing
- `overlay_transparency` (float): Transparency level

### POST `/cut-out-georeferenced-map`
Cut georeferenced map sections based on geometry input.

**Parameters:**
- `geometry` (string): Input geometry
- `map_type` (string): Map source
- `buffer` (float): Buffer distance in meters

### Utility Endpoints
- `GET /` - Service information
- `GET /health` - Health status
- `GET /sessions` - List active sessions
- `GET /download/{session_id}/{file_path}` - Download files
- `DELETE /sessions/{session_id}` - Clean up session

## Map Types

| Type | Description | Best For |
|------|-------------|----------|
| `osm` | OpenStreetMap | General mapping |
| `bgt-omtrek` | Dutch BGT outlines | Technical analysis |
| `luchtfoto` | Dutch aerial photography | High-resolution imagery |
| `brta` | BRT-A topographic | Topographic mapping |
| `bag` | Building footprints | Building analysis |
| `bgt-bg-bag` | BGT + buildings | Combined base maps |

## AI Models

### KAZE Features (Schematic Maps)
- **Use Case**: Technical diagrams, CAD drawings, schematic maps
- **Algorithm**: Nonlinear diffusion-based feature detection
- **Fallbacks**: ORB → SIFT for robustness
- **Preprocessing**: Edge enhancement for line-based content

### GIM+Roma (Aerial Imagery)
- **Use Case**: Satellite images, aerial photography, luchtfoto
- **Architecture**: DINOv2 backbone + global-to-local matching
- **Features**: Uncertainty estimation, coarse-to-fine correspondence
- **Performance**: State-of-the-art deep learning matcher
- **Requirements**: ~4GB RAM (CPU) or ~4GB VRAM (GPU)

## Quality Assessment

**Matching Quality:**
- **Excellent**: ≥30% inlier ratio
- **Good**: ≥20% inlier ratio  
- **Fair**: ≥10% inlier ratio
- **Poor**: <10% inlier ratio

## Configuration

### Environment Variables
```bash
PORT=8000                    # Service port
UPLOAD_DIR=uploads          # Upload directory
OUTPUT_DIR=outputs          # Output directory
OMP_NUM_THREADS=1           # OpenMP threads
MKL_NUM_THREADS=1           # MKL threads
```

### Model Setup
Models are automatically downloaded on first use:
- **GIM+Roma**: ~1.2GB download from Hugging Face
- **Caching**: Persistent Docker volumes prevent re-downloads
- **CPU Compatibility**: Mac systems automatically use CPU-only mode

## Technical Architecture

```
services/
├── feature_matching_service.py    # Core matching logic
├── map_cutting_service.py         # Map cutting & tiles  
├── georeferencing_service.py      # World files & GeoTIFF
├── visualization_service.py       # Overlays & results
└── image_processing_service.py    # Image optimization

utils/
├── coordinate_utils.py            # RD New ↔ Web Mercator
├── geometry_utils.py              # Shapely buffering
├── file_utils.py                  # Session management
└── validation_utils.py            # Input validation

third_party/
├── roma_minimal.py                # GIM+Roma implementation
└── RoMa/                          # Model repository
```

## Data Flow

1. **Input**: Geometry + map type OR image pairs
2. **Buffer Testing**: Multiple buffer sizes tested automatically
3. **Map Cutting**: Tiles downloaded and stitched
4. **Feature Matching**: KAZE or GIM+Roma based on map type
5. **Georeferencing**: World files created, GeoTIFF output
6. **Quality Check**: Inlier ratio assessment
7. **Output**: Best results with comparison data

## Development

### Local Setup
```bash
pip install -r requirements.txt
python app.py
```

### Key Dependencies
- **FastAPI + Uvicorn**: REST API
- **OpenCV**: KAZE/ORB/SIFT features
- **PyTorch**: GIM+Roma deep learning
- **Shapely**: Geometric operations
- **rasterio**: GeoTIFF support
- **pyproj**: Coordinate transformations

## Model Citations

```bibtex
@article{shen2024gim,
  title={GIM: Learning Generalizable Image Matcher From Internet Videos},
  author={Shen, Xuelun and others},
  journal={ICLR}, year={2024}
}

@article{edstedt2023roma,
  title={RoMa: A Lightweight Multi-Camera 3D Depth Estimation Framework},
  author={Edstedt, Johan and others},
  year={2023}
}

@article{alcantarilla2012kaze,
  title={KAZE Features},
  author={Alcantarilla, Pablo F and others},
  journal={ECCV}, year={2012}
}
```

## License

MIT License. Third-party models (GIM+Roma, DINOv2) maintain their respective licenses.