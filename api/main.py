from fastapi import FastAPI, HTTPException, Query
import httpx
from typing import Dict, Any, List, Optional
import os
from datetime import date

# Import our models and constants
from models import BordcodeCategory, Province
from constants import DEFAULT_N8N_WEBHOOK_URL

# Initialize FastAPI app
app = FastAPI(
    title="Traffic decree processing API",
    description="API to process traffic decrees",
    version="1.0.0"
)

# N8N webhook URL - loaded from environment variable set by docker-compose
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", DEFAULT_N8N_WEBHOOK_URL)
print(f"Using N8N_WEBHOOK_URL: {N8N_WEBHOOK_URL}")

@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the N8N traffic decree workflow"}

@app.post("/trigger-n8n-workflow")
async def trigger_n8n_workflow(
    start_date: date = Query(..., description="The start date from which traffic decrees should be processed in YYYY-MM-DD format."),
    end_date: date = Query(..., description="The end date until which traffic decrees should be processed in YYYY-MM-DD format."),
    bordcode_categories: Optional[str] = Query(
        None, 
        description="Filter by bordcode categories. Input must be one of the following values: 'A, B, C, D, E, F, J, K, L'"
    ),
    provinces: Optional[str] = Query(
        None, 
        description="Filter by Dutch provinces. Use comma-separated values: 'utrecht,gelderland' or 'utrecht, gelderland'"
    ),
    gemeenten: Optional[str] = Query(
        None, 
        description="Filter by municipalities. Use comma-separated values: 'amsterdam,rotterdam' or 'amsterdam, rotterdam'"
    ), 
    exclude_keywords: Optional[str] = Query(
        None, 
        description="Exclude keywords from the search. Use comma-separated values: 'parkeerplaats,laadpaal'. The more relevant keywords are excluded, the faster the model will work. Advice is to use at least parkeerplaats, laadpaal, parkeerverbod, parkeervergunning, parkeerregime, parkeermogelijkheden, parkeervoorzieningen, parkeersituatie, parkeersituaties, parkeerplaatsen, parkeerplaatsvoorzieningen."
    )
):
    """
    Starts the N8N workflow to process traffic decisions for a given date range with optional filtering.
    Resulting besluiten are stored in the postgres database.
    You can follow the progress of the workflow in the n8n web interface.
    
    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format  
        bordcode_categories: Optional list of bordcode categories (A, B, C, D, E, F, J, K, L). 
                           Includes decisions if metadata contains ANY of these letters.
        provinces: Optional list of Dutch provinces (case-insensitive)
        gemeenten: Optional list of municipalities (case-insensitive)
        
    Important: 
        If you set the name of a province, it will only return besluiten from the province, not the municipalities in that province.
        If you set the name of a municipality, it will only return besluiten from the municipality, not the province.
    """


    # Create payload from parameters
    payload_data = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    
    # Parse and validate comma-separated inputs
    if bordcode_categories:
        # Split by comma and clean up whitespace
        categories = [cat.strip().upper() for cat in bordcode_categories.split(",") if cat.strip()]
        # Validate each category
        valid_categories = []
        for cat in categories:
            try:
                validated_cat = BordcodeCategory(cat)
                valid_categories.append(validated_cat.value)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid bordcode category: '{cat}'. Valid values are: A, B, C, D, E, F, G, H, J, K, L"
                )
        payload_data["bordcode_categories"] = valid_categories
    
    if provinces:
        # Split by comma and clean up whitespace
        province_list = [prov.strip().lower() for prov in provinces.split(",") if prov.strip()]
        # Validate each province
        valid_provinces = []
        for prov in province_list:
            try:
                # Try to match with enum values
                validated_prov = Province(prov)
                valid_provinces.append(validated_prov.value)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid province: '{prov}'. Valid values are: {', '.join([p.value for p in Province])}"
                )
        payload_data["provinces"] = valid_provinces
    
    if gemeenten:
        # Split by comma and clean up whitespace (no validation needed for gemeente names)
        gemeente_list = [gem.strip().lower() for gem in gemeenten.split(",") if gem.strip()]
        payload_data["gemeenten"] = gemeente_list
    
    if exclude_keywords:
        # Split by comma and clean up whitespace
        exclude_keywords_list = [keyword.strip().lower() for keyword in exclude_keywords.split(",") if keyword.strip()]
        payload_data["exclude_keywords"] = exclude_keywords_list

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(N8N_WEBHOOK_URL, json=payload_data)
            response.raise_for_status()  # Raises an exception for 4XX/5XX responses
            return {"message": "N8N traffic decree workflow triggered successfully", "n8n_response": response.json()}
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Error communicating with N8N: {exc}"
            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Error response from N8N: {exc.response.text}"
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 