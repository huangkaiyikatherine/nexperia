DATASET=nexperia_split
DATA_ROOT='\home\kaiyihuang\nexperia\new_data'
ARCH=resnet34
LR=0.1
LR_SCHEDULE='cosine'
EPOCHS=200
BATCH_SIZE=32
LOSS=sat
ALPHA=0.9
ES=150
NOISE_RATE=0.02
NOISE_TYPE='corrupted_label'
TRAIN_SETS='trainval'
VAL_SETS='test_set'
MOD='bad_1'
EXP_NAME=${DATASET}/${ARCH}_${LOSS}_${MOD}_m${ALPHA}_p${ES}_lr${LR}_${LR_SCHEDULE}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='0'
TEN_CROP=true
DATA_ROOT='/home/kaiyihuang/nexperia/new_data'
FREQ=50

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs
mkdir -p logs/${DATASET}


### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main.py --arch ${ARCH} --loss ${LOSS} \
        --sat-alpha ${ALPHA} --sat-es ${ES} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --noise-rate ${NOISE_RATE} --noise-type ${NOISE_TYPE} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} --ten-crop ${TEN_CROP} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} \
        --mod ${MOD} \
        >> ${LOG_FILE} 2>&1
