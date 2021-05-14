DATASET=nexperia_merge
# DATA_ROOT='/home/kaiyihuang/nexperia/batch_3/all'
DATA_ROOT='/home/kaiyihuang/nexperia/new_data'
DATA_ROOT_1='/home/kaiyihuang/nexperia/new_data'
DATA_ROOT_2='/home/kaiyihuang/nexperia/batch_3'
DATA_ROOT_3='/import/home/share/SourceData/DownSampled/Jul'
ARCH=resnet50
LR=0.001
LR_SCHEDULE='cosine'
EPOCHS=200
BATCH_SIZE=32
LOSS=ce
TRAIN_SETS='trainval'
VAL_SETS='test_set'
CROP='center'
PRETRAINED='false'
EXP_NAME=${DATASET}/${ARCH}_pretrained_${PRETRAINED}_crop_${CROP}_${LOSS}_lr${LR}_${LR_SCHEDULE}_epoch${EPOCHS}_batchsize_${BATCH_SIZE}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='7'
FREQ=500
SAVE_FREQ=0

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs
mkdir -p logs/${DATASET}


### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main_general.py --arch ${ARCH} --loss ${LOSS} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --data-root-1 ${DATA_ROOT_1} --data-root-2 ${DATA_ROOT_2} --data-root-3 ${DATA_ROOT_3} \
        --crop ${CROP} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} --save-freq ${SAVE_FREQ} \
        >> ${LOG_FILE} 2>&1
