#!/bin/bash
IN_DIR="real_images"
OUT_DIR="masked_images"

mkdir -p $OUT_DIR
for SHIRT_DIR in "$IN_DIR"/*
do
    for IMG_FILENAME in "$SHIRT_DIR"/*
    do
        mkdir -p "${SHIRT_DIR/$IN_DIR/"$OUT_DIR"}"
        if [[ $IMG_FILENAME == *"color"* ]]
        then
            MASK_IMG_FILENAME="${IMG_FILENAME/$IN_DIR/"$OUT_DIR"}"
            MASK_IMG_FILENAME="${MASK_IMG_FILENAME/color_/"rgb-"}"
            backgroundremover -i "$IMG_FILENAME" -o "$MASK_IMG_FILENAME"
        fi
    done
done

python3 process_real_data.py