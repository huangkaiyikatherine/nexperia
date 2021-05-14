DATASET=nexperia_eval
# DATA_ROOT='/home/kaiyihuang/nexperia/batch_3/all'
DATA_ROOT='/import/home/share/from_Nexperia_April2021/Jan2021'
DATA_ROOT_1='/home/kaiyihuang/nexperia/new_data'
DATA_ROOT_2='/home/kaiyihuang/nexperia/batch_3'
ARCH=resnet50
LR=0.001
LR_SCHEDULE='cosine'
EPOCHS=0
BATCH_SIZE=32
LOSS=ce
VAL_SET='Jan2021_path2label.csv'
CROP='center'
PRETRAINED='false'
CHECKPOINT='/home/kaiyihuang/giants_shoulder/self-adaptive-training-master/ckpts/nexperia_train/resnet50_sat_bad_1_m0.9_p50_lr0.001_cosine_/checkpoint_latest.tar'
LOAD_MODEL='best_model_1'
EXP_NAME=${DATASET}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='1'
FREQ=500
SAVE_FREQ=10
EVAL=True

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs
mkdir -p logs/${DATASET}


### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main_eval.py --arch ${ARCH} --loss ${LOSS} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} --data-root-1 ${DATA_ROOT_1} --data-root-2 ${DATA_ROOT_2} --crop ${CROP} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} \
        --val-set ${VAL_SET} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        --print-freq ${FREQ} --save-freq ${SAVE_FREQ} \
        --evaluate ${EVAL} \
        --checkpoint ${CHECKPOINT} --load-model ${LOAD_MODEL} \
        >> ${LOG_FILE} 2>&1
