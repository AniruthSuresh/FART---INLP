#!/bin/bash

# Create datasets directory if it doesn't exist
mkdir -p datasets

# Download IMDB dataset
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -P datasets/

# Extract the dataset
tar -xzf datasets/aclImdb_v1.tar.gz -C datasets/

# Remove the tar file to save space
rm datasets/aclImdb_v1.tar.gz

echo "IMDB dataset downloaded and extracted to datasets/aclImdb/"