DATASET=nexperia_split
DATA_ROOT='\home\kaiyihuang\nexperia\new_data'
ARCH=resnet34
LR=0.01
LR_SCHEDULE='cosine'
EPOCHS=200
BATCH_SIZE=32
LOSS=sat_multi_es
ALPHA=0.9
ES1=150
ES2=150
ES3=150
ES4=150
ES5=150
ES6=150
ES7=150
ES8=150
ES9=150
ES10=150
NOISE_RATE=0.02
NOISE_TYPE='corrupted_label'
TRAIN_SETS='trainval'
VAL_SETS='test_set'
EXP_NAME=${DATASET}/${ARCH}_${LOSS}_m${ALPHA}_p1${ES1}_p2${ES2}_p3${ES3}_p4${ES4}_p5${ES5}_p6${ES6}_p7${ES7}_p8${ES8}_p9${ES9}_p10${ES10}_lr${LR}_${LR_SCHEDULE}_$1
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
        --sat-alpha ${ALPHA} --sat-es ${ES} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --noise-rate ${NOISE_RATE} --noise-type ${NOISE_TYPE} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --train-sets ${TRAIN_SETS} --val-sets ${VAL_SETS} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} \
        --sat-es1 ${ES1} --sat-es2 ${ES2} --sat-es3 ${ES3} --sat-es4 ${ES4} --sat-es5 ${ES5} \
        --sat-es6 ${ES6} --sat-es7 ${ES7} --sat-es8 ${ES8} --sat-es9 ${ES9} --sat-es10 ${ES10} \
        >> ${LOG_FILE} 2>&1
