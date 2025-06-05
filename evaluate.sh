#!/bin/bash

# Create necessary directories
mkdir -p AgentProjectData/EvaluationResults

export PYTHONPATH=src
python src/evaluate.py

echo "Evaluation completed! Check AgentProjectData/EvaluationResults/ for results."