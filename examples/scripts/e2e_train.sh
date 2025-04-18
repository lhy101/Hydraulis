MODEL_SIZE=${1:-"32b"}
GLOBAL_BATCH_SIZE=-1 
GLOBAL_TOKEN_NUM=${2:-100000} # use gtn instead of gbs
MAX_SEQ_LEN=${3:-32768}
SERVER_ADDR=${7:-"0.0.0.0"} # your server address here
SERVER_PORT=${8:-"23333"}
HOST_FILE_PATH=${9:-"./hostfile/host.yaml"}
ENV_FILE_PATH=${10:-"./scripts/env.sh"}

# example
# 64 GPUs 32B CommonCrawl
# the "case study (Figure 12)" setup in our paper
NUM_GPUS=64
MODEL_SIZE=32b
GLOBAL_TOKEN_NUM=100000
MAX_SEQ_LEN=32768
# CommonCrawl
ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/data.json 
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

# Dynamic Heterogeneous Strategies
MULTI_CP_TP_PP_LIST="[[(1, 16, 1), (1, 16, 1), (1, 16, 1), (1, 16, 1)], [(1, 16, 1), (1, 8, 3), (1, 8, 3)], [(1, 16, 1), (1, 8, 1), (1, 8, 1), (1, 8, 1), (1, 8, 1), (1, 8, 1), (1, 8, 1)], [(1, 16, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1)], [(1, 8, 3), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1)], [(1, 8, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1)], [(1, 4, 2), (1, 4, 2), (1, 4, 2), (1, 4, 2), (1, 4, 2), (1, 4, 2), (1, 4, 2), (1, 4, 2)]]"

# Two-stage sequence assignment
BATCHING_METHOD=4

if [ "${MODEL_SIZE}" = "7b" ]; then
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=11008
    NUM_HEADS=32
elif [ "${MODEL_SIZE}" = "13b" ]; then
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    FFN_HIDDEN_SIZE=13824
    NUM_HEADS=40
elif [ "${MODEL_SIZE}" = "32b" ]; then
    NUM_LAYERS=60
    HIDDEN_SIZE=6656 
    FFN_HIDDEN_SIZE=17920
    NUM_HEADS=64
else
    echo the model should be 7b/13b/32b for test.
    exit 0
fi

echo num_gpus=${NUM_GPUS}, global_token_num = ${GLOBAL_TOKEN_NUM}, max_seq_len = ${MAX_SEQ_LEN}

# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/llama${MODEL_SIZE}_gpus${NUM_GPUS}_gtn${GLOBAL_TOKEN_NUM}_msl${MAX_SEQ_LEN}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

# compute-sanitizer can be added in front of python3 to check illegal mem access bug
CMD="python3 -u e2e_train.py \
--batching_method $BATCHING_METHOD \
--multi_cp_tp_pp_list \"${MULTI_CP_TP_PP_LIST}\" \
--global_batch_size $GLOBAL_BATCH_SIZE \
--global_token_num $GLOBAL_TOKEN_NUM \
--max_seq_len $MAX_SEQ_LEN \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--epochs 4 \
--steps 40 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS}"

source ${ENV_FILE_PATH}
if [ ${NUM_GPUS} -gt 8 ]; then
python3 ../../python_refactor/hydraulis/rpc/pssh_start.py \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
else
python3 ../../python_refactor/hydraulis/rpc/pssh_start.py \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
fi