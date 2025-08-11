#!/bin/bash

# Ollama Initialization Script
# This script starts Ollama and pulls the required models

echo "🚀 Starting Ollama initialization..."

# Start Ollama in the background
ollama serve &
OLLAMA_PID=$!

echo "⏳ Waiting for Ollama to be ready..."
# Wait for Ollama to be ready (using a simple sleep approach)
sleep 10

# Get model from environment variable (default to llama3.2:1b)
MODEL=${OLLAMA_MODEL:-llama3.2:1b}

echo "✅ Ollama should be ready! Pulling model: $MODEL"

# Pull the specified model
ollama pull $MODEL

echo "🎉 Model pulled successfully!"

# List available models for confirmation
echo "📋 Available models:"
ollama list

echo "🔄 Keeping Ollama running..."
# Keep the script running to maintain the Ollama process
wait $OLLAMA_PID