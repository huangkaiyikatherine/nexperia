DATASET=nexperia_split
DATA_ROOT='/home/kaiyihuang/nexperia/new_data'
DATA_ROOT_1='/home/kaiyihuang/nexperia/new_data'
DATA_ROOT_2='/home/kaiyihuang/nexperia/batch_3'
ARCH=resnet50
LR=0.001
LR_SCHEDULE='cosine'
EPOCHS=2
BATCH_SIZE=32
LOSS=sat
ALPHA=0.9
ES=1
MOD='bad_boost'
# NOISE_RATE=0.02
# NOISE_TYPE='corrupted_label'
TRAIN_SETS='trainval'
VAL_SETS='test_set'
EXP_NAME=${DATASET}/${ARCH}_${LOSS}_${MOD}_m${ALPHA}_p${ES}_lr${LR}_${LR_SCHEDULE}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='1'
FREQ=500

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs
mkdir -p logs/${DATASET}


### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main.py --arch ${ARCH} --loss ${LOSS} \
        --sat-alpha ${ALPHA} --sat-es ${ES} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} --data-root-1 ${DATA_ROOT_1} --data-root-2 ${DATA_ROOT_2} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} \
        --mod ${MOD} \
        >> ${LOG_FILE} 2>&1
