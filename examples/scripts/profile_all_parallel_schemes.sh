CP_VALUES=(1 2 4 8)
TP_VALUES=(1 2 4 8)
PP_VALUES=(1 2 4 8)

for CP in "${CP_VALUES[@]}"; do
  for TP in "${TP_VALUES[@]}"; do
    for PP in "${PP_VALUES[@]}"; do
      if (( CP * TP * PP > 64 )); then
        continue
      fi
      if (( CP * TP * PP < 1 )); then
        continue
      fi
      # 定义 EXP_FILE 的路径
      EXP_FILE="./experiments/cp${CP}_tp${TP}_pp${PP}.txt"
      # 调用现有脚本
      bash scripts/profile.sh "CP" "$TP" "$PP" "$EXP_FILE"
    done
  done
done
