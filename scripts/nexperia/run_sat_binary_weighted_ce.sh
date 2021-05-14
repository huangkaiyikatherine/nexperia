DATASET=nexperia_split
DATA_ROOT='\home\kaiyihuang\nexperia\new_data'
ARCH=resnet34
LR=0.001
LR_SCHEDULE='cosine'
EPOCHS=200
BATCH_SIZE=32
LOSS=sat_binary_weighted_ce
ALPHA=0.9
ES=90
MOD='bad_1'
TRAIN_SETS='trainval'
VAL_SETS='test_set'
EL_1=100
EL_2=125
EL_3=125
EL_4=200
EL_5=200
EL_6=200
EL_7=100
EL_8=125
EL_9=90
EL_10=100
CE_MOMENTUM=1
EXP_NAME=${DATASET}/${ARCH}_${LOSS}_${MOD}_m${ALPHA}_p${ES}_el1_${EL_1}_el2_${EL_2}_el3_${EL_3}_el4_${EL_4}_el5_${EL_5}_el6_${EL_6}_el7_${EL_7}_el8_${EL_8}_el9_${EL_9}_el10_${EL_10}_cem${CE_MOMENTUM}_lr${LR}_${LR_SCHEDULE}_epoch${EPOCHS}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='0'
DATA_ROOT='/home/kaiyihuang/nexperia/new_data'
FREQ=100

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs
mkdir -p logs/${DATASET}


### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main.py --arch ${ARCH} --loss ${LOSS} \
        --sat-alpha ${ALPHA} --sat-es ${ES} --mod ${MOD} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} \
        --el1 ${EL_1} --el2 ${EL_2} --el3 ${EL_3} --el4 ${EL_4} --el5 ${EL_5} \
        --el6 ${EL_6} --el7 ${EL_7} --el8 ${EL_8} --el9 ${EL_9} --el10 ${EL_10} \
        --ce-momentum ${CE_MOMENTUM} \
        >> ${LOG_FILE} 2>&1

