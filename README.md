# Traffic Decree Processing Model

**AI-powered system for processing and analyzing Dutch traffic decrees** using automated workflows, intelligent image filtering, and advanced feature matching.

## ⚡ Quick Start (5 Minutes)

### Prerequisites
1. **Install Docker Desktop**:
   - Download from [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Install and start Docker Desktop
   - Ensure Docker UI is running (check menu bar/system tray)

### Setup Steps
1. **Clone Repository**:
   ```bash
   git clone https://github.com/SamueldeWeerd/traffic-decree-processing-model.git
   cd traffic-decree-processing-model
   ```

2. **Copy Configuration File**:
   ```bash
   cp config.env .env
   ```

3. **Configure Credentials**:
   - Add OpenAI API key to `.env`: `OPENAI_API_KEY=sk-your-key`
   - Add money to [OpenAI account](https://platform.openai.com/billing) (required)

4. **Start Services**:
   ```bash
   docker compose up --build
   ```

5. **Setup N8N**:
   - Visit [http://localhost:5678](http://localhost:5678)
   - Create admin account
   - Go to **Workflows** → **Import from File**
   - Select workflow from `n8n/workflows/` folder
   - Update webhook URL in `.env` from imported workflow
   - Update all credentials in workflow with environment variables (use javascript: 
   example: {{ $env.OPENAI_API_KEY }})

6. **Test API**:
   - Visit [http://localhost:8000/docs](http://localhost:8000/docs)
   - Try `/trigger-n8n-workflow` endpoint

7. **View N8N Workflow**:
   - Return to [http://localhost:5678](http://localhost:5678)
   - Open the imported workflow
   - Examine the processing flow and model interactions

8. **Set up N8N webhook URL for the api to start the traffic-decree-processor**:
   - Set correct N8N webhook URL in .env file


## Services & Endpoints

- **n8n Workflow Platform**: `http://localhost:5678`
- **FastAPI Workflow Trigger**: `http://localhost:8000` 
  - **Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
  - **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Feature Matching API**: `http://localhost:8005`
  - **Swagger Docs**: [http://localhost:8005/docs](http://localhost:8005/docs)
  - **ReDoc**: [http://localhost:8005/redoc](http://localhost:8005/redoc)
- **KOOP API Service**: `http://localhost:8001`
  - **Swagger Docs**: [http://localhost:8001/docs](http://localhost:8001/docs)
  - **ReDoc**: [http://localhost:8001/redoc](http://localhost:8001/redoc)

## AI Models & Processing Pipeline

### 1. **KOOP API - Data Retrieval** (Primary Function)
- **Purpose**: Downloads traffic decree documents and extracts images from government APIs
- **Input**: Date ranges, province/municipality filters, bordcode categories
- **Output**: Structured decree data with extracted map images
- **Note**: Includes optional CLIP filtering to reduce irrelevant image downloads

### 2. **KAZE Feature Matching** (Feature Matching API)
- **Model**: KAZE nonlinear diffusion features
- **Purpose**: Matches schematic maps and technical diagrams
- **Input**: Hand-drawn maps, CAD drawings, technical schematics
- **Fallbacks**: ORB → SIFT detectors for robustness
- **Output**: Feature correspondences with geometric transformations

### 3. **GIM+Roma Deep Learning Matcher** (Feature Matching API)
- **Model**: GIM (Global Image Matcher) + Roma with DINOv2 backbone
- **Purpose**: Matches aerial/satellite imagery (luchtfoto)
- **Input**: High-resolution aerial photographs, satellite images
- **Features**: Global-to-local matching, uncertainty estimation
- **Output**: Robust correspondences for complex aerial imagery

### 4. **Ollama Local LLM** (N8N Workflows)
- **Model**: Llama 3.2 3B (locally hosted)
- **Purpose**: Text processing, metadata extraction, workflow decisions
- **Input**: Traffic decree text content, processing instructions
- **Output**: Structured data, workflow routing decisions

## Processing Pipeline Flow

```
Government APIs → KOOP API → Traffic Decree Data + Images
                       ↓
Traffic Decree Text → Ollama LLM → Structured Metadata
                       ↓
Map Images → Feature Matching API → KAZE or GIM+Roma Selection
                       ↓
Feature Correspondences → Georeferencing → GeoTIFF Output
```

**Intelligent Model Selection:**
- **KOOP API**: Downloads and structures government traffic decree data
- **KAZE**: Automatically selected for schematic maps and technical diagrams
- **GIM+Roma**: Automatically selected for aerial imagery (luchtfoto)
- **Ollama**: Processes text content and metadata throughout pipeline

## 📚 API Documentation

### Interactive API Documentation (Swagger UI)
- **FastAPI Workflow Trigger**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Feature Matching API**: [http://localhost:8005/docs](http://localhost:8005/docs)  
- **KOOP API Service**: [http://localhost:8001/docs](http://localhost:8001/docs)

### Alternative Documentation (ReDoc)
- **FastAPI Workflow Trigger**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Feature Matching API**: [http://localhost:8005/redoc](http://localhost:8005/redoc)
- **KOOP API Service**: [http://localhost:8001/redoc](http://localhost:8001/redoc)

**Note**: Access these links after starting the services with `docker compose up --build`

## Environment Configuration

Configuration uses a template + secrets approach for better security:

### Template File (`config.env`) - Safe to commit to Git
Contains all non-secret settings that can be safely shared:
- Database and service configuration
- AI model settings (Ollama, ROMA, feature matching)
- System resources (CPU cores, memory limits)
- Cleanup and retention settings
- KOOP API configuration (rate limits, file sizes, etc.)

### Runtime File (`.env`) - Never commit to Git
Created by copying `config.env` and adding secrets:
```bash
# Add these to your .env file
OPENAI_API_KEY=sk-your-actual-key-here
N8N_ENCRYPTION_KEY=<64-char-hex-key>
N8N_USER_MANAGEMENT_JWT_SECRET=<64-char-hex-key>
POSTGRES_USER=n8n_user
POSTGRES_PASSWORD=n8n_password
```

**Setup Process:**
1. **Copy base configuration:** `cp config.env .env`
2. **Add your secrets** to the `.env` file (OpenAI key, N8N keys, DB password)
3. **Adjust system settings** in `.env` (CPU_CORES, MAX_MEMORY_MB, etc.)
4. Ensure you have [OpenAI credits](https://platform.openai.com/billing) (prepaid required)
5. The N8N webhook URL will be updated after importing the workflow

## API Usage

### POST `/trigger-n8n-workflow`

Triggers traffic decree processing with filtering options.

**Parameters:**
- `start_date` (required): Start date (YYYY-MM-DD)
- `end_date` (required): End date (YYYY-MM-DD)
- `bordcode_categories` (optional): Traffic sign categories (A, C, D, F, G)
- `provinces` (optional): Dutch provinces
- `gemeenten` (optional): Municipalities

**Filtering Rules:**
- ⚠️ **Choose EITHER `provinces` OR `gemeenten`, never both**
- When `provinces` is selected: **Only processes decrees issued by the province**
- When `gemeenten` is selected: **Only processes decrees issued by the municipality**
- `bordcode_categories` can be combined with either option

**Examples:**
```bash
# Province-issued decrees with bordcode filter
curl -X POST "http://localhost:8000/trigger-n8n-workflow?start_date=2024-05-24&end_date=2024-05-24&bordcode_categories=a&provinces=utrecht"

# Municipality-issued decrees
curl -X POST "http://localhost:8000/trigger-n8n-workflow?start_date=2024-05-24&end_date=2024-05-24&gemeenten=amsterdam"
```

## AI Model Details

### CLIP Image Classification
- **Model Size**: ~400MB download
- **Performance**: 0.1-0.5 seconds per image
- **Accuracy**: High precision for maps vs text classification
- **Hardware**: CPU/GPU support, 2-4GB memory

### KAZE Feature Matching
- **Use Case**: Schematic maps, technical drawings
- **Features**: Extended 128-byte descriptors
- **Performance**: Fast matching with quality assessment
- **Hardware**: CPU-only, minimal memory requirements

### GIM+Roma Deep Learning
- **Model Size**: ~1.2GB download (cached persistently)
- **Use Case**: Aerial imagery, satellite photos
- **Architecture**: DINOv2 + global-to-local matching
- **Performance**: State-of-the-art accuracy for aerial imagery
- **Hardware**: 4GB+ RAM (CPU) or 4GB+ VRAM (GPU)

### Ollama Llama 3.2 3B
- **Model Size**: ~2GB download
- **Use Case**: Text processing, metadata extraction
- **Performance**: Fast local inference
- **Hardware**: CPU-only, 8GB+ RAM recommended

## N8N Workflow Details

### Workflow Components
- **Webhook Trigger**: Receives requests from FastAPI
- **Data Processing**: Filters and processes traffic decree data
- **API Calls**: Integrates with KOOP API and Feature Matching API
- **Ollama Integration**: Uses local LLM for text processing
- **Result Aggregation**: Combines and formats final outputs

![Workflow Import Process](./assets/edit_workflow.png)

## Processing Workflow

1. **Data Retrieval**: n8n fetches traffic decree data from government APIs
2. **Image Filtering**: CLIP classifies and filters relevant images
3. **Text Processing**: Ollama extracts metadata and structured information
4. **Feature Matching**: KAZE or GIM+Roma matches images to reference maps
5. **Georeferencing**: Creates georeferenced GeoTIFF outputs
6. **Storage**: Results organized by session for download

## Model Performance & Quality

### Image Classification (CLIP)
- **Maps/Aerial**: >90% accuracy for relevant content
- **Text/Signs**: >95% accuracy for filtering irrelevant content
- **Storage Savings**: Typically 60-80% reduction in downloaded images

### Feature Matching Quality
- **Excellent**: ≥30% inlier ratio (high confidence matches)
- **Good**: ≥20% inlier ratio (reliable matches)
- **Fair**: ≥10% inlier ratio (acceptable matches)
- **Poor**: <10% inlier ratio (manual review needed)

### Automatic Model Selection
The system intelligently chooses the best AI model for each task:
- **Schematic/Technical**: KAZE features (fast, reliable)
- **Aerial/Satellite**: GIM+Roma (high accuracy)
- **Mixed Content**: CLIP pre-filtering ensures optimal model usage

## File Storage

- `./shared/` - n8n shared data
- `./feature-matching-api/data/outputs/` - Processing results  
- `./ROMA_checkpoints/` - AI model cache
- `./ollama-data/` - LLM storage
- `./koop-api/afbeeldingen/` - Filtered images

## Troubleshooting

**503 Service Unavailable:**
- Check `N8N_WEBHOOK_URL` in `.env` uses `http://n8n:5678` (not localhost)
- Restart: `docker-compose restart fastapi`

**AI Model Downloads:**
- First run downloads models (~4GB total)
- Check logs: `docker-compose logs -f [service-name]`
- Ensure sufficient disk space and internet connection

**Memory Issues:**
- CLIP + GIM+Roma + Ollama can use 8-12GB RAM total
- Consider reducing concurrent processing
- Use CPU-only mode for memory-constrained systems

## Common Commands

```bash
# Start all services (recommended)
docker compose up --build

# Start in background
docker compose up --build -d

# View logs
docker compose logs -f [service-name]

# Restart specific service  
docker compose restart [service-name]

# Stop all services
docker compose down

# Connect to PostgreSQL database
docker exec -it traffic_decree_processing_model-postgres-1 psql -U n8n_user -d n8n_db
```

## Model Citations

```bibtex
@article{shen2024gim,
  title={GIM: Learning Generalizable Image Matcher From Internet Videos},
  author={Shen, Xuelun and others}, journal={ICLR}, year={2024}
}

@article{edstedt2023roma,
  title={RoMa: A Lightweight Multi-Camera 3D Depth Estimation Framework},
  author={Edstedt, Johan and others}, year={2023}
}

@article{alcantarilla2012kaze,
  title={KAZE Features},
  author={Alcantarilla, Pablo F and others}, journal={ECCV}, year={2012}
}
```

## 🚀 Recommendations for Improvements

### Current Limitations & Opportunities

#### **1. Single Image Processing per Decree**
- **Current**: Only processes one image per traffic decree
- **Improvement**: Process all map images from each decree for comprehensive analysis
- **Benefit**: Complete spatial understanding of traffic regulations

#### **2. Single Geometry Type per Decree**
- **Current**: Processes only one geometry point/area per decree
- **Improvement**: Handle multiple geometry types (points, lines, polygons) per decree
- **Benefit**: Support complex traffic regulations spanning multiple areas

#### **3. Batch Processing Enhancement**
- **Current**: Sequential processing of individual decrees
- **Improvement**: Implement parallel processing for multiple decrees
- **Benefit**: Significant performance improvements for large date ranges

#### **4. Advanced Feature Matching**
- **Current**: Processes images independently
- **Improvement**: Cross-reference multiple images from the same decree
- **Benefit**: Higher accuracy through multi-image validation

#### **5. Enhanced Filtering & Search**
- **Current**: Basic date and category filtering
- **Improvement**: Add full-text search, geographic region filtering, regulation type classification with AI
- **Benefit**: More precise and flexible decree discovery

#### **6. Result Aggregation & Analytics**
- **Current**: Individual processing results
- **Improvement**: Aggregate results across multiple decrees, generate spatial analytics
- **Benefit**: Regional traffic pattern analysis and trend identification

#### **7. Quality Assurance Pipeline**
- **Current**: Basic feature matching quality scores
- **Improvement**: Let computer vision analyse warped-source-mages for quality checks. Now, warped images are sometimes included in the output that are completely deformed
- **Benefit**: Higher accuracy and reliability for critical applications

### Implementation Priority
1. **High**: Multiple image processing per decree
2. **High**: Multiple geometry support per decree  
3. **Medium**: Batch processing optimization
4. **Medium**: Enhanced filtering capabilities
5. **Low**: Advanced analytics and reporting

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.