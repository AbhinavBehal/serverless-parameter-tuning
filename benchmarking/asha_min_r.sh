#!/usr/bin/env bash

current_folder="${BASH_SOURCE%/*}"
results_file="${current_folder}/results/asha_min_r.csv"
temp_file="${current_folder}/.temp"

mkdir -p "${current_folder}/results" && touch ${results_file}

echo "Workers,Min_R,Max_R,Reduction_Factor,Early_Stopping_Rounds,Time,Score,Configs_Evaluated" > ${results_file}

for i in 1 2 4 8 16 32 64 128 256; do
    echo ${i}
    workers=100
    min_r=${i}
    max_r=256
    early_stopping_rounds=${max_r}
    reduction_factor=2

    params="{\"n_workers\": ${workers}, \"min_r\": ${min_r}, \"max_r\": ${max_r},\
     \"reduction_factor\": ${reduction_factor}, \"early_stopping_rounds\": ${early_stopping_rounds},\"cv\": 3}"

    python "${current_folder}/../main.py" -a asha -p "${params}" > ${temp_file}
    score=`grep -oP "(?<=Best score: ).+" ${temp_file}`
    time=`grep -oP "(?<=Took: )[\\d\\.]+" ${temp_file}`
    configs=`grep -oP "(?<=Total configs evaluated: ).+" ${temp_file}`
    echo "${workers},${min_r},${max_r},${reduction_factor},${early_stopping_rounds},${time},${score},${configs}" >> ${results_file}
done

rm -f ${temp_file} &> /dev/null
