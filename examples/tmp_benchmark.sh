#!/bin/bash

DATA_DIR=/data
RES_DIR=/output/t-3dgs

# on-the-go dataset
DATASET_NAME="on-the-go"
FPS=2
RESOLUTION=8

for SCENE in fountain mountain corner patio_high spot patio
do
	python train.py \
		-s="${DATA_DIR}/${DATASET_NAME}/${SCENE}" \
		-m="${RES_DIR}/${DATASET_NAME}/${SCENE}" \
        --fps=$FPS \
		--eval \
        --resolution=$RESOLUTION
done

# robustnerf dataset
DATASET_NAME="robustnerf"
FPS=2
RESOLUTION=8

for SCENE in android crab1 crab2 statue yoda
do
	python train.py \
		-s="${DATA_DIR}/${DATASET_NAME}/${SCENE}" \
		-m="${RES_DIR}/${DATASET_NAME}/${SCENE}" \
        --fps=$FPS \
		--eval \
        --resolution=$RESOLUTION
done

# t-3dgs dataset
DATASET_NAME="t-3dgs"
FPS=8

for SCENE in lab1 lab2 library anti_stress office
do
    if [ "$SCENE" = "office" ]; then
        RESOLUTION=4
    else
        RESOLUTION=8
    fi
	python train.py \
		-s="${DATA_DIR}/${DATASET_NAME}/${SCENE}" \
		-m="${RES_DIR}/${DATASET_NAME}/${SCENE}" \
        --fps=$FPS \
		--eval \
        --resolution=$RESOLUTION
done
