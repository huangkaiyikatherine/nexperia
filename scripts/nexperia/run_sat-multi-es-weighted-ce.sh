DATASET=nexperia_split
DATA_ROOT='\home\kaiyihuang\nexperia\new_data'
ARCH=resnet34
LR=0.01
LR_SCHEDULE='cosine'
EPOCHS=200
BATCH_SIZE=32
LOSS=sat_multi_es_weighted_ce
ALPHA=0.9
ES1=90
ES2=90
ES3=200
ES4=90
ES5=90
ES6=90
ES7=90
ES8=90
ES9=90
ES10=90
EL_1=90
EL_2=90
EL_3=200
EL_4=90
EL_5=200
EL_6=90
EL_7=90
EL_8=90
EL_9=90
EL_10=90
CE_MOMENTUM=1
NOISE_RATE=0.02
NOISE_TYPE='corrupted_label'
TRAIN_SETS='trainval'
VAL_SETS='test_set'
EXP_NAME=${DATASET}/${ARCH}_${LOSS}_m${ALPHA}_p1_${ES1}_p2_${ES2}_p3_${ES3}_p4_${ES4}_p5_${ES5}_p6_${ES6}_p7_${ES7}_p8_${ES8}_p9_${ES9}_p10_${ES10}_el1_${EL_1}_el2_${EL_2}_el3_${EL_3}_el4_${EL_4}_el5_${EL_5}_el6_${EL_6}_el7_${EL_7}_el8_${EL_8}_el9_${EL_9}_el10_${EL_10}_cem${CE_MOMENTUM}_lr${LR}_${LR_SCHEDULE}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='1'
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
        --sat-alpha ${ALPHA} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --noise-rate ${NOISE_RATE} --noise-type ${NOISE_TYPE} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} \
        --sat-es1 ${ES1} --sat-es2 ${ES2} --sat-es3 ${ES3} --sat-es4 ${ES4} --sat-es5 ${ES5} \
        --sat-es6 ${ES6} --sat-es7 ${ES7} --sat-es8 ${ES8} --sat-es9 ${ES9} --sat-es10 ${ES10} \
        --el1 ${EL_1} --el2 ${EL_2} --el3 ${EL_3} --el4 ${EL_4} --el5 ${EL_5} \
        --el6 ${EL_6} --el7 ${EL_7} --el8 ${EL_8} --el9 ${EL_9} --el10 ${EL_10} \
        --ce-momentum ${CE_MOMENTUM} \
        >> ${LOG_FILE} 2>&1
