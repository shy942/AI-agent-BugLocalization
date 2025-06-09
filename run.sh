#!/bin/bash
export PYTHONPATH=src
#python src/main.py > log2.txt 2>&1
mkdir -p logs/terminal_logs
python src/KEYBERT.py > logs/terminal_logs/keybert_log.txt 2>&1 &   
python src/NLP.py > logs/terminal_logs/NLP_log.txt 2>&1 &  
python src/reason.py > logs/terminal_logs/reason_log.txt 2>&1 &   
# python src/main.py > log2.txt 2>&1
mkdir -p logs/terminal_logs
mkdir -p logs/parallel_logs
python src/NLP.py > logs/terminal_logs/NLP_log.txt 2>&1 &  
python src/KEYBERT.py > logs/terminal_logs/keybert_log.txt 2>&1 &   
python src/reason.py > logs/terminal_logs/reason_log.txt 2>&1 &   

wait  # Wait for all scripts to finish

echo "All processing completed. Check logs/terminal_logs/ for output logs."
echo "Check AgentProjectData/BM25andFAISS/ for generated indexes."
echo "Check AgentProjectData/ConstructedQueries/ for generated queries."
echo "Check AgentProjectData/SearchResults/ for search results."

