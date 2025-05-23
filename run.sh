#!/bin/bash
export PYTHONPATH=src
mkdir -p logs/terminal_logs
python src/KEYBERT.py > logs/terminal_logs/keybert_log.txt 2>&1 &   
python src/NLP.py > logs/terminal_logs/NLP_log.txt 2>&1 &  
python src/reason.py > logs/terminal_logs/reason_log.txt 2>&1 &   

wait  # Wait for both scripts to finish

