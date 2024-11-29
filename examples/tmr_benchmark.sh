#!/bin/bash

DATA_DIR=/data
DATASET_NAME=t-3dgs
MODEL_DIR=/output/t-3dgs_refined
FPS=8
MASKS_DIR=/output/tmr

for SCENE in lab1 lab2 library anti_stress office
do
    if [ "$SCENE" = "office" ]; then
        RESOLUTION=4
    else
        RESOLUTION=8
    fi
	python train.py \
		-s="${DATA_DIR}/${DATASET_NAME}/${SCENE}" \
		-m="${MODEL_DIR}/${DATASET_NAME}/${SCENE}" \
		--masks="${MASKS_DIR}/${DATASET_NAME}/${SCENE}/binary_masks" \
		--disable_transient \
		--fps=$FPS \
		--eval \
        --resolution=$RESOLUTION
done
