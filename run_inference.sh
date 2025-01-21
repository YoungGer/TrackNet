#!/bin/bash

# 不带插值的版本
python infer_on_video.py \
    --model_path ./model/tracknet_model_best.pt \
    --video_path ./input_video.mp4 \
    --video_out_path ./output_video.mp4 \
    --batch_size 2

# # 带插值的版本
# python infer_on_video.py \
#     --model_path ./model/tracknet_model_best.pt \
#     --video_path ./input_video.mp4 \
#     --video_out_path ./output_video_with_extrapolation.mp4 \
#     --batch_size 2 \
#     --extrapolation 