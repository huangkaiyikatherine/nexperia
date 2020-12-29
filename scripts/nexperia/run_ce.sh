DATASET=nexperia_split
DATA_ROOT='\home\kaiyihuang\nexperia\new_data'
<<<<<<< HEAD
ARCH=resnet18
LR=0.005
LR_SCHEDULE='cosine'
EPOCHS=1
=======
ARCH=resnet34
LR=0.01
LR_SCHEDULE='cosine'
EPOCHS=200
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb
BATCH_SIZE=32
LOSS=ce
ALPHA=0.9
ES=150
NOISE_RATE=0.02
NOISE_TYPE='corrupted_label'
TRAIN_SETS='trainval'
VAL_SETS='test_set'
<<<<<<< HEAD
CROP='center'
EXP_NAME=${DATASET}/${ARCH}_crop_${CROP}_${LOSS}_lr${LR}_${LR_SCHEDULE}_epoch${EPOCHS}_batchsize_${BATCH_SIZE}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='3'
DATA_ROOT='/home/kaiyihuang/nexperia/new_data'
FREQ=100
=======
PRETRAINED=true
FEATURE_EXTRACTION=true
EXP_NAME=${DATASET}/${ARCH}_pt_${PRETRAINED}_fe_${FEATURE_EXTRACTION}_${LOSS}_lr${LR}_${LR_SCHEDULE}_epoch${EPOCHS}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='3'
TEN_CROP=true
DATA_ROOT='/home/kaiyihuang/nexperia/new_data'
FREQ=50
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs
mkdir -p logs/${DATASET}


### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main.py --arch ${ARCH} --loss ${LOSS} \
        --sat-alpha ${ALPHA} --sat-es ${ES} \
<<<<<<< HEAD
        --dataset ${DATASET} --data-root ${DATA_ROOT} --crop ${CROP} \
        --noise-rate ${NOISE_RATE} --noise-type ${NOISE_TYPE} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} \
=======
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --noise-rate ${NOISE_RATE} --noise-type ${NOISE_TYPE} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} --ten-crop ${TEN_CROP} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} \
        --pretrained ${PRETRAINED} --feature-extraction ${FEATURE_EXTRACTION} \
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb
        >> ${LOG_FILE} 2>&1
