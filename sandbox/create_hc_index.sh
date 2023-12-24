#!/bin/bash

BASE_INDEX_NAME=msmarco_splade_100k
INDEX_NAME=${BASE_INDEX_NAME}_bp
INDEX_ROOT=index
PISA_BIN="../../pisa-score-cache/build-rel/bin"
EVAL_SET=msmarco_splade_train_5k_1000.trec
TRAIN_QUERIES=/home/clownland/Datasets/msmarco/queries_splade_5k.train.pisa
COUNT_K=100

set -x
set -e

$PISA_BIN/evaluate_queries \
    -e block_simdbp -i $INDEX_ROOT/inverted/${INDEX_NAME}.simdbp \
    -w $INDEX_ROOT/wand/${INDEX_NAME}.wand --tokenizer whitespace \
    --terms $INDEX_ROOT/forward/${BASE_INDEX_NAME}.termlex \
    -q ${TRAIN_QUERIES} --weighted -k 1000 -a maxscore -s quantized -j 6 \
    --documents $INDEX_ROOT/forward/${INDEX_NAME}.doclex > $EVAL_SET

HC_INDEX=${INDEX_NAME}_hc_${COUNT_K}

$PISA_BIN/access_count_ordering -t $EVAL_SET \
                                -d $INDEX_ROOT/forward/${INDEX_NAME}.doclex \
                                -k $COUNT_K \
                                > ${HC_INDEX}.mapping

$PISA_BIN/reorder-docids \
    -c $INDEX_ROOT/inverted/$INDEX_NAME \
    -o $INDEX_ROOT/inverted/${HC_INDEX} \
    --documents $INDEX_ROOT/forward/${INDEX_NAME}.doclex \
    --reordered-documents $INDEX_ROOT/forward/${HC_INDEX}.doclex \
    -j 6 --from-mapping ${HC_INDEX}.mapping

$PISA_BIN/create_wand_data -c $INDEX_ROOT/inverted/${HC_INDEX} \
                           -o $INDEX_ROOT/wand/${HC_INDEX}.wand \
                           -s quantized -b 40

$PISA_BIN/compress_inverted_index \
    -c $INDEX_ROOT/inverted/${HC_INDEX} \
    -o $INDEX_ROOT/inverted/${HC_INDEX}.simdbp -e block_simdbp
