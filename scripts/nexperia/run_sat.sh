DATASET=nexperia_split
DATA_ROOT='\home\kaiyihuang\nexperia\new_data'
ARCH=resnet34
<<<<<<< HEAD
LR=0.01
=======
LR=0.1
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb
LR_SCHEDULE='cosine'
EPOCHS=200
BATCH_SIZE=32
LOSS=sat
ALPHA=0.9
<<<<<<< HEAD
ES=90
=======
ES=150
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb
NOISE_RATE=0.02
NOISE_TYPE='corrupted_label'
TRAIN_SETS='trainval'
VAL_SETS='test_set'
<<<<<<< HEAD
MOD='bad_boost'
=======
MOD='bad_1'
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb
EXP_NAME=${DATASET}/${ARCH}_${LOSS}_${MOD}_m${ALPHA}_p${ES}_lr${LR}_${LR_SCHEDULE}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='0'
<<<<<<< HEAD
DATA_ROOT='/home/kaiyihuang/nexperia/new_data'
FREQ=100
=======
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
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --noise-rate ${NOISE_RATE} --noise-type ${NOISE_TYPE} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
<<<<<<< HEAD
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
=======
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} --ten-crop ${TEN_CROP} \
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} \
        --mod ${MOD} \
        >> ${LOG_FILE} 2>&1
