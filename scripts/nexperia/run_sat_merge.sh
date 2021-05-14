DATASET=nexperia_merge
DATA_ROOT='/home/kaiyihuang/nexperia/new_data'
DATA_ROOT_1='/home/kaiyihuang/nexperia/new_data'
DATA_ROOT_2='/home/kaiyihuang/nexperia/batch_3'
DATA_ROOT_3='/import/home/share/SourceData/DownSampled/Jul'
ARCH=resnet50
LR=0.001
LR_SCHEDULE='cosine'
EPOCHS=200
BATCH_SIZE=32
LOSS=sat_fl
FL_LAMBDA=0.7
FL_ALPHA=1
FL_GAMMA=2
ALPHA=0.9
ES=90
MOD='bad_boost'
# NOISE_RATE=0.02
# NOISE_TYPE='corrupted_label'
TRAIN_SET='/import/home/share/SourceData/DownSampled/Jul_train_kfold.txt'
VAL_SET='/import/home/share/SourceData/DownSampled/Jul_crossval_kfold.txt'
TEST_SET='/import/home/share/SourceData/DownSampled/Jul_test_kfold.txt'
EXP_NAME=${DATASET}/${ARCH}_${LOSS}_${MOD}_m${ALPHA}_p${ES}_lr${LR}_${LR_SCHEDULE}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='1'
FREQ=500

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs
mkdir -p logs/${DATASET}


### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main_general.py --arch ${ARCH} --loss ${LOSS} \
        --sat-alpha ${ALPHA} --sat-es ${ES} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --data-root-1 ${DATA_ROOT_1} --data-root-2 ${DATA_ROOT_2} --data-root-3 ${DATA_ROOT_3} \
        --train-set ${TRAIN_SET} --val-set ${VAL_SET} --test-set ${TEST_SET} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --fl-lambda ${FL_LAMBDA} --fl-alpha ${FL_ALPHA} --fl-gamma ${FL_GAMMA} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} \
        --mod ${MOD} \
        >> ${LOG_FILE} 2>&1
