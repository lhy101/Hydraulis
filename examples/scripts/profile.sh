CP=${1:-2}
TP=${2:-2}
PP=${3:-2}
EXP_FILE=${4:-"./experiments/tp2_pp2.txt"}
DP=1

NUM_LAYERS=60
HIDDEN_SIZE=6656
NUM_HEADS=64
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
FFN_HIDDEN_SIZE=17920
SERVER_ADDR="0.0.0.0" # need your server address
SERVER_PORT="23462"
HOST_FILE_PATH="./hostfiles/host01.yaml"
ENV_FILE_PATH="./scripts/env.sh"

NUM_GPUS=$(expr $CP \* $TP \* $PP)
DCP=${CP}
CP_LIST="["
for ((i=1; i<=DP; i++)); do
	if [ $i -ne 1 ]; then
		CP_LIST="$CP_LIST,"
	fi
	CP_LIST="$CP_LIST$CP"
done
CP_LIST="$CP_LIST]"
RECOMPUTE_LAYERS="[]"

echo run exp: cp=${CP}, tp=${TP}, pp=${PP}, num_gpus=${NUM_GPUS} 
LOG_FOLDER=logs/exp_cp${CP}_tp${TP}_pp${PP}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

python ./ds_parallel_config/generate_gpt_3d_config.py \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $DP \
	--cp $CP \
	--tp $TP \
	--pp $PP \
	--zero \
	--recompute_layers $RECOMPUTE_LAYERS

EXP_DIR=$(dirname "$EXP_FILE")
if [ ! -d "$EXP_DIR" ]; then
  mkdir -p "$EXP_DIR"
fi
if [ ! -e "$EXP_FILE" ]; then
	> "$EXP_FILE"
fi

START_SEQ=128
for i in $(seq ${START_SEQ} 128 65536); do

content=$(<"$EXP_FILE")
length=${#content}
if [[ "${content:length-2:1}" == ":" ]]; then
	echo "run exp: already OOM"
    break
fi
if [[ "${content:length-1:1}" == ":" ]]; then
	echo "run exp: already OOM"
    break
fi
echo "run exp: seq_len = ${i}"
echo "seq len = ${i}:" >> "$EXP_FILE"
SEQ_LEN=${i}
CMD="python3 -u profile.py \
--num_strategy=1 \
--ds_parallel_config ds_parallel_config/homo/dcp${DCP}_tp${TP}_pp${PP}.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--micro_batch_size $MICRO_BATCH_SIZE \
--global_seq_len $SEQ_LEN \
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
--ngpus ${NUM_GPUS} \
--cp_list \"${CP_LIST}\" \
--exp_file ${EXP_FILE}"

source ${ENV_FILE_PATH}
if [ ${NUM_GPUS} -gt 8 ]; then
python3 ../../../python_refactor/hydraulis/rpc/pssh_start_exp.py \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
else
python3 ../../../python_refactor/hydraulis/rpc/pssh_start_exp.py \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
fi

done
