#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# API base URL
API_URL="http://localhost:8005"

# Test files
SOURCE_IMAGE="data/test_images/source_image_luchtfoto.png"
DEST_IMAGE="data/test_images/dest_image.png"
DEST_PGW="data/test_images/destination.pgw"

#geometry for luchtfoto
GEOMETRY='{
    "type": "Point",
    "coordinates": [121559.8068878073, 534396.834230677]
}'


# #Test geometry (RD New coordinates)
# GEOMETRY='{
#     "type": "Point",
#     "coordinates": [197985, 368028]
# }'

#Test geometry for url endpoint
GEOMETRY='{
    "type": "Point",
    "coordinates": [199600, 409900]
}'

# Test geometry for utrecht
# GEOMETRY='{
#     "type": "Point",
#     "coordinates": [
#         [137766, 457273]
#     ]
# }'

# GEOMETRY for boxtel POINT(150751 398981)
# GEOMETRY='{
#     "type": "Point",
#     "coordinates": [
#         [150751, 398981]
#     ]
# }'


echo -e "\n${BLUE}=== Testing API Endpoints ===${NC}\n"

# Function to print section header
print_header() {
    echo -e "\n${BLUE}=== Testing $1 ===${NC}\n"
}

# Function to check if a file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: File $1 not found${NC}"
        exit 1
    fi
}

# Check if required files exist
check_file "$SOURCE_IMAGE"
check_file "$DEST_IMAGE"
check_file "$DEST_PGW"

# Test variables
TRAFFIC_DECREE_ID="GMB-100"
MAP_TYPES=("brta")
BUFFER=20

# Function to test an endpoint with multiple map types
test_endpoint() {
    local endpoint=$1
    local description=$2
    local map_type=$3
    shift 3
    local args=("$@")

    print_header "${description} (${map_type})"
    "${args[@]}" | json_pp
    echo -e "\n${GREEN}Press Enter to continue...${NC}"
    read
}

# # 2. Test map cutting endpoint
# for map_type in "${MAP_TYPES[@]}"; do
#     test_endpoint "/cut-out-georeferenced-map" "Map Cutting Endpoint" "$map_type" \
#         curl -X POST "${API_URL}/cut-out-georeferenced-map" \
#         -H "accept: application/json" \
#         -H "Content-Type: application/x-www-form-urlencoded" \
#         -d "geometry=$(echo $GEOMETRY | jq -c .)" \
#         -d "map_type=${map_type}" \
#         -d "buffer=${BUFFER}" \
#         -d "output_format=json" \
#         -d "traffic_decree_id=${TRAFFIC_DECREE_ID}"
# done

# # 3. Test cutout-and-match endpoint
# for map_type in "${MAP_TYPES[@]}"; do
#     test_endpoint "/cutout-and-match" "Cutout and Match Endpoint" "$map_type" \
#         curl -X POST "${API_URL}/cutout-and-match" \
#         -H "accept: application/json" \
#         -F "source_image=@${SOURCE_IMAGE}" \
#         -F "geometry=$(echo $GEOMETRY | jq -c .)" \
#         -F "map_type=${map_type}" \
#         -F "overlay_transparency=0.6" \
#         -F "output_format=json" \
#         -F "traffic_decree_id=${TRAFFIC_DECREE_ID}"
# done

# 4. Test cutout-and-match-with-url endpoint
for map_type in "${MAP_TYPES[@]}"; do
    test_endpoint "/cutout-and-match-with-url" "Cutout and Match with URL Endpoint" "$map_type" \
        curl -X POST "${API_URL}/cutout-and-match-with-url" \
        -H "accept: application/json" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "image_url=https://zoek.officielebekendmakingen.nl/gmb-2024-219045-1.jpg" \
        -d "geometry=$(echo $GEOMETRY | jq -c .)" \
        -d "map_type=${map_type}" \
        -d "overlay_transparency=0.6" \
        -d "output_format=json" \
        -d "traffic_decree_id=${TRAFFIC_DECREE_ID}"
done

echo -e "\n${BLUE}=== All Tests Completed ===${NC}\n"

# Optional: List all sessions
print_header "Listing All Sessions"
curl -X GET "${API_URL}/sessions" | json_pp

# Ask if user wants to cleanup sessions
echo -e "\n${GREEN}Do you want to cleanup all sessions? (y/n)${NC}"
read cleanup

if [ "$cleanup" = "y" ]; then
    echo -e "\n${BLUE}Cleaning up sessions...${NC}"
    # Get all session IDs and delete them
    sessions=$(curl -s "${API_URL}/sessions" | jq -r '.sessions[].session_id')
    for session in $sessions; do
        echo "Deleting session $session..."
        curl -X DELETE "${API_URL}/sessions/${session}"
    done
    echo -e "\n${GREEN}All sessions cleaned up${NC}"
fi