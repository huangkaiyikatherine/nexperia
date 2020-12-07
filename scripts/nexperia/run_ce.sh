DATASET=nexperia_split
DATA_ROOT='\home\kaiyihuang\nexperia\new_data'
ARCH=resnet34
LR=0.01
LR_SCHEDULE='cosine'
EPOCHS=200
BATCH_SIZE=32
LOSS=ce
ALPHA=0.9
ES=150
NOISE_RATE=0.02
NOISE_TYPE='corrupted_label'
TRAIN_SETS='trainval'
VAL_SETS='test_set'
PRETRAINED=true
FEATURE_EXTRACTION=true
EXP_NAME=${DATASET}/${ARCH}_pt_${PRETRAINED}_fe_${FEATURE_EXTRACTION}_${LOSS}_lr${LR}_${LR_SCHEDULE}_epoch${EPOCHS}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='3'
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
        --pretrained ${PRETRAINED} --feature-extraction ${FEATURE_EXTRACTION} \
        >> ${LOG_FILE} 2>&1
