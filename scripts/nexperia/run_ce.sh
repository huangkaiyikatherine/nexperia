DATASET=nexperia_split
DATA_ROOT='\home\kaiyihuang\nexperia\batch_3/all'
ARCH=resnet18
LR=0.001
LR_SCHEDULE='cosine'
EPOCHS=50
BATCH_SIZE=256
LOSS=ce
ALPHA=0.9
ES=150
NOISE_RATE=0.02
NOISE_TYPE='corrupted_label'
TRAIN_SETS='trainval'
VAL_SETS='test_set'
CROP='center'
PRETRAINED='true'
EXP_NAME=${DATASET}/${ARCH}_pretrained_${PRETRAINED}_crop_${CROP}_${LOSS}_lr${LR}_${LR_SCHEDULE}_epoch${EPOCHS}_batchsize_${BATCH_SIZE}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='0'
DATA_ROOT='/home/kaiyihuang/nexperia/new_data'
FREQ=200

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs
mkdir -p logs/${DATASET}


### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main.py --arch ${ARCH} --loss ${LOSS} \
        --sat-alpha ${ALPHA} --sat-es ${ES} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} --crop ${CROP} \
        --noise-rate ${NOISE_RATE} --noise-type ${NOISE_TYPE} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} \
        --pretrained ${PRETRAINED} \
        >> ${LOG_FILE} 2>&1
