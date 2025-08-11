"""
Constants for the traffic decree processing API
"""

# Valid Dutch provinces (lowercase for case-insensitive matching)
VALID_PROVINCES = [
    "drenthe",
    "flevoland", 
    "friesland",
    "gelderland",
    "groningen",
    "limburg",
    "noord-brabant",
    "noord-holland",
    "overijssel",
    "utrecht",
    "zeeland",
    "zuid-holland"
]

# Valid bordcode categories
VALID_BORDCODE_CATEGORIES = ["A", "C", "D", "F", "G"]

# API Configuration
DEFAULT_N8N_WEBHOOK_URL = "http://n8n:5678/webhook/your-webhook-id-here"