#!/bin/bash

DATA_DIR=/data
DATASET_NAME=t-3dgs
MODEL_DIR=/output/t-3dgs
ITERATION=7000

for SCENE in lab1 lab2 library anti_stress office
do
    if [ "$SCENE" = "office" ]; then
        RESOLUTION=4
    else
        RESOLUTION=8
    fi
	python extract_tmr_prompt.py \
		--source_path="${DATA_DIR}/${DATASET_NAME}/${SCENE}" \
		--model_path="${MODEL_DIR}/${DATASET_NAME}/${SCENE}" \
        --iteration=$ITERATION \
        --resolution=$RESOLUTION

    python prepare_images_for_tmr.py \
		-s="${DATA_DIR}/${DATASET_NAME}/${SCENE}/images" \
		-t="${DATA_DIR}/${DATASET_NAME}/${SCENE}/images_tmr" \
        --resolution=$RESOLUTION
done
