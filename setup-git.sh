#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up Git repository...${NC}"

# Initialize Git repository if not already initialized
if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}✅ Git repository initialized${NC}"
else
    echo -e "${GREEN}✓ Git repository already exists${NC}"
fi

# Create .gitignore file
echo -e "${BLUE}📝 Creating .gitignore...${NC}"
cat > .gitignore << EOL
# Environment files
.env
.env.*
!config.env

# Data directories
ROMA_checkpoints/
ollama-data/
huggingface-cache/

# Python
__pycache__/
*.py[cod]
*$py.class
.Python
.env/
.venv/
venv/
ENV/

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs
*.log
logs/
log/

# System
.DS_Store
Thumbs.db

# Docker
.docker/
docker-compose.override.yml
EOL
echo -e "${GREEN}✅ .gitignore created${NC}"

# Add all files except those in .gitignore
echo -e "${BLUE}📦 Adding files to Git...${NC}"
git add .

# Show status
echo -e "${BLUE}📊 Current Git status:${NC}"
git status

echo -e "\n${BLUE}🎉 Setup complete! Next steps:${NC}"
echo -e "1. Review the files to be committed using 'git status'"
echo -e "2. Make your first commit using: git commit -m 'Initial commit'"
echo -e "3. Add your remote repository: git remote add origin <repository-url>"
echo -e "4. Push your code: git push -u origin main"