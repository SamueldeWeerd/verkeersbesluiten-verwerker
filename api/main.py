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
    start_date: date = Query(..., description="The start date from which traffic decrees should be processed."),
    end_date: date = Query(..., description="The end date until which traffic decrees should be processed."),
    bordcode_categories: Optional[str] = Query(
        None, 
        description="Filter by bordcode categories. Use comma-separated values: 'A,C,D, F, G' or 'A, C, D, F, G'"
    ),
    provinces: Optional[str] = Query(
        None, 
        description="Filter by Dutch provinces. Use comma-separated values: 'utrecht,gelderland' or 'utrecht, gelderland'"
    ),
    gemeenten: Optional[str] = Query(
        None, 
        description="Filter by municipalities. Use comma-separated values: 'amsterdam,rotterdam' or 'amsterdam, rotterdam'"
    )
):
    """
    Receives parameters and forwards them to an N8N webhook that starts the traffic decree workflow.
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
                    detail=f"Invalid bordcode category: '{cat}'. Valid values are: A, C, D, F, G"
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