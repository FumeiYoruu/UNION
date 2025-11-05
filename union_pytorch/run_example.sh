#!/bin/bash

# Example script to train and evaluate UNION models

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}UNION PyTorch Example Scripts${NC}"
echo -e "${BLUE}================================${NC}"

# Example 1: Train BERT on ROCStories
echo -e "\n${GREEN}Example 1: Train BERT on ROCStories${NC}"
echo "Command:"
echo "python train.py \\"
echo "    --task_name train \\"
echo "    --model_type bert \\"
echo "    --model_name bert-base-uncased \\"
echo "    --data_dir ../Data/ROCStories \\"
echo "    --output_dir ./output/bert_roc \\"
echo "    --dataset_mode roc \\"
echo "    --max_seq_length 200 \\"
echo "    --train_batch_size 8 \\"
echo "    --num_train_epochs 3 \\"
echo "    --learning_rate 2e-5 \\"
echo "    --warmup_steps 500 \\"
echo "    --save_steps 1000"

# Example 2: Train Longformer on WritingPrompts (16k context)
echo -e "\n${GREEN}Example 2: Train Longformer on WritingPrompts (16k context)${NC}"
echo "Command:"
echo "python train.py \\"
echo "    --task_name train \\"
echo "    --model_type longformer \\"
echo "    --model_name allenai/longformer-base-16384 \\"
echo "    --data_dir ../Data/WritingPrompts \\"
echo "    --output_dir ./output/longformer_wp \\"
echo "    --dataset_mode wp \\"
echo "    --max_seq_length 2048 \\"
echo "    --train_batch_size 4 \\"
echo "    --gradient_accumulation_steps 2 \\"
echo "    --num_train_epochs 3 \\"
echo "    --learning_rate 2e-5"

# Example 3: Prediction
echo -e "\n${GREEN}Example 3: Predict and Evaluate${NC}"
echo "Command:"
echo "python predict.py \\"
echo "    --task_name pred \\"
echo "    --model_type bert \\"
echo "    --model_name bert-base-uncased \\"
echo "    --data_dir ../Data/ROCStories \\"
echo "    --output_dir ./output/predictions \\"
echo "    --dataset_mode roc \\"
echo "    --max_seq_length 200 \\"
echo "    --init_checkpoint ./output/bert_roc/best-epoch3-step12000 \\"
echo "    --eval_batch_size 16"

# Example 4: Training with advanced features
echo -e "\n${GREEN}Example 4: Train with Multi-layer Pooling + Reconstruction${NC}"
echo "Command:"
echo "python train.py \\"
echo "    --task_name train \\"
echo "    --model_type longformer \\"
echo "    --data_dir ../Data/WritingPrompts \\"
echo "    --output_dir ./output/longformer_advanced \\"
echo "    --dataset_mode wp \\"
echo "    --max_seq_length 2048 \\"
echo "    --use_all_layers \\"
echo "    --use_reconstruction \\"
echo "    --reconstruction_weight 0.1 \\"
echo "    --train_batch_size 4 \\"
echo "    --gradient_accumulation_steps 4"

# Example 5: Training on Award-winning dataset with reconstruction
echo -e "\n${GREEN}Example 5: Train on Award-winning Stories with Reconstruction${NC}"
echo "Note: First prepare the data with: python ../Data/prepare_award_winning.py"
echo "Command:"
echo "python train.py \\"
echo "    --task_name train \\"
echo "    --model_type bert \\"
echo "    --model_name bert-base-uncased \\"
echo "    --data_dir ../Data/Award-winning \\"
echo "    --output_dir ./output/bert_award \\"
echo "    --dataset_mode award \\"
echo "    --max_seq_length 512 \\"
echo "    --use_reconstruction \\"
echo "    --reconstruction_weight 0.1 \\"
echo "    --train_batch_size 4 \\"
echo "    --num_train_epochs 5 \\"
echo "    --learning_rate 2e-5 \\"
echo "    --warmup_steps 100"

echo -e "\n${BLUE}================================${NC}"
echo -e "${BLUE}To run an example, copy and paste the command above${NC}"
echo -e "${BLUE}================================${NC}"
