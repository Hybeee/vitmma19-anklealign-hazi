#! /bin/bash
set -e

timestamp=$(date +%Y%m%d_%H%M%S)

mkdir -p data outputs
mkdir -p outputs/$timestamp
mkdir -p outputs/$timestamp/plots

echo "Running data preparation..."
python data_preparing.py --timestamp $timestamp

echo "Running data processing..."
python data_processing.py --timestamp $timestamp

echo "Running model training..."
python train.py --timestamp $timestamp

echo "Running evaluation..."
python evaluation.py --timestamp $timestamp

echo "Pipeline finished successfully."