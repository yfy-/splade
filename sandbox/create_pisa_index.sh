#!/bin/bash

DOCS=splade_msmarco_centroid_unq_3.tsv
INDEX_NAME=splade_msmarco_centroid_unq_3
INDEX_ROOT=index
PISA_BIN="../../pisa/build-rel/bin"

cat $DOCS | $PISA_BIN/parse_collection -o $INDEX_ROOT/forward/$INDEX_NAME \
                                       -b 6000 -j 6 --tokenizer whitespace

$PISA_BIN/invert -i $INDEX_ROOT/forward/$INDEX_NAME \
                 -o $INDEX_ROOT/inverted/$INDEX_NAME -j 6 --batch-size 6000

$PISA_BIN/reorder-docids \
    -c $INDEX_ROOT/inverted/$INDEX_NAME \
    -o $INDEX_ROOT/inverted/${INDEX_NAME}_bp \
    --documents $INDEX_ROOT/forward/${INDEX_NAME}.doclex \
    --reordered-documents $INDEX_ROOT/forward/${INDEX_NAME}_bp.doclex -d 10 \
    -j 6 --bp

$PISA_BIN/create_wand_data -c $INDEX_ROOT/inverted/${INDEX_NAME}_bp \
                           -o $INDEX_ROOT/wand/${INDEX_NAME}_bp.wand \
                           -s quantized -b 40

$PISA_BIN/compress_inverted_index \
    -c $INDEX_ROOT/inverted/${INDEX_NAME}_bp \
    -o $INDEX_ROOT/inverted/${INDEX_NAME}_bp.simdbp -e block_simdbp
