DATASET=nexperia_split
DATA_ROOT='\home\kaiyihuang\nexperia\new_data'
ARCH=resnet34
LR=0.01
LR_SCHEDULE='cosine'
EPOCHS=200
BATCH_SIZE=32
LOSS=binary_weighted_ce
ALPHA=0.9
ES=150
NOISE_RATE=0.02
NOISE_TYPE='corrupted_label'
TRAIN_SETS='trainval'
VAL_SETS='test_set'
EL_1=200
EL_2=200
EL_3=200
EL_4=200
EL_5=200
EL_6=70
EL_7=200
EL_8=35
EL_9=0
EL_10=200
<<<<<<< HEAD
CE_MOMENTUM=0.1
EXP_NAME=${DATASET}/${ARCH}_${LOSS}_el1_${EL_1}_el2_${EL_2}_el3_${EL_3}_el4_${EL_4}_el5_${EL_5}_el6_${EL_6}_el7_${EL_7}_el8_${EL_8}_el9_${EL_9}_el10_${EL_10}_momentum${CE_MOMENTUM}_lr${LR}_${LR_SCHEDULE}_epoch${EPOCHS}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='0'
=======
CE_MOMENTUM=1
EXP_NAME=${DATASET}/${ARCH}_${LOSS}_el1_${EL_1}_el2_${EL_2}_el3_${EL_3}_el4_${EL_4}_el5_${EL_5}_el6_${EL_6}_el7_${EL_7}_el8_${EL_8}_el9_${EL_9}_el10_${EL_10}_momentum${CE_MOMENTUM}_lr${LR}_${LR_SCHEDULE}_epoch${EPOCHS}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='1'
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb
DATA_ROOT='/home/kaiyihuang/nexperia/new_data'
FREQ=50
### [90, 100, 200, 150, 90, 70, 200, 130, 0, 25]

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
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} \
        --el1 ${EL_1} --el2 ${EL_2} --el3 ${EL_3} --el4 ${EL_4} --el5 ${EL_5} \
        --el6 ${EL_6} --el7 ${EL_7} --el8 ${EL_8} --el9 ${EL_9} --el10 ${EL_10} \
        --ce-momentum ${CE_MOMENTUM} \
        >> ${LOG_FILE} 2>&1
