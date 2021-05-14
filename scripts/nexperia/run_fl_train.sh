DATASET=nexperia_train
DATA_ROOT='/import/home/share/from_Nexperia_April2021/Nex_trainingset'
DATA_ROOT_1='/home/kaiyihuang/nexperia/new_data'
DATA_ROOT_2='/home/kaiyihuang/nexperia/batch_3'
ARCH=resnet50
LR=0.001
LR_SCHEDULE='cosine'
EPOCHS=200
BATCH_SIZE=32
LOSS=fl
FL_ALPHA=1
FL_GAMMA=2
TRAIN_SET='Nex_trainingset_train.txt'
VAL_SET='Nex_trainingset_val.txt'
TEST_SET='Nex_trainingset_test.txt'
CROP='center'
PRETRAINED='false'
EXP_NAME=${DATASET}/${ARCH}_pretrained_${PRETRAINED}_crop_${CROP}_${LOSS}_fl_alpha${FL_ALPHA}_fl_gamma${FL_GAMMA}_lr${LR}_${LR_SCHEDULE}_epoch${EPOCHS}_batchsize_${BATCH_SIZE}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='6'
FREQ=500
SAVE_FREQ=10

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs
mkdir -p logs/${DATASET}


### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main_train.py --arch ${ARCH} --loss ${LOSS} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} --data-root-1 ${DATA_ROOT_1} --data-root-2 ${DATA_ROOT_2} --crop ${CROP} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --fl-alpha ${FL_ALPHA} --fl-gamma ${FL_GAMMA} \
        --train-set ${TRAIN_SET} --val-set ${VAL_SET} --test-set ${TEST_SET} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} --save-freq ${SAVE_FREQ} \
        >> ${LOG_FILE} 2>&1
