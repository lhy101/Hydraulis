TP_VALUES=(1 2 4 8)
PP_VALUES=(1 2 4 8)

for TP in "${TP_VALUES[@]}"; do
  for PP in "${PP_VALUES[@]}"; do
    if (( TP * PP > 64 )); then
      continue
    fi
    if (( TP * PP < 1 )); then
      continue
    fi
    # 定义 EXP_FILE 的路径
    EXP_FILE="./experiments/tp${TP}_pp${PP}.txt"
    # 调用现有脚本
    bash scripts/profile.sh "$TP" "$PP" "$EXP_FILE"
  done
done
