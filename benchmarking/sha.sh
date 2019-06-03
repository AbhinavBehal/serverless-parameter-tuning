#!/usr/bin/env bash

current_folder="${BASH_SOURCE%/*}"
results_file="${current_folder}/results/sha.csv"
temp_file="${current_folder}/.temp"

mkdir -p "${current_folder}/results" && touch ${results_file}

echo "Configs,Min_R,Max_R,Reduction_Factor,Time,Score" > ${results_file}

for i in 2 4 8 16 32 64 128 256 512 1024; do
    echo ${i}

    min_r=1
    max_r=${i}
    reduction_factor=2
    params="{\"n_workers\": -1, \"n_configs\": ${i}, \"min_r\": ${min_r}, \"max_r\": ${max_r},\
     \"reduction_factor\": ${reduction_factor}, \"cv\": 3}"

    python "${current_folder}/../main.py" -a sha -p "${params}" > ${temp_file}
    score=`grep -oP "(?<=Best score: ).+" ${temp_file}`
    time=`grep -oP "(?<=Took: )[\\d\\.]+" ${temp_file}`
    echo "${i},${min_r},${max_r},${reduction_factor},${time},${score}" >> ${results_file}
done

rm -f ${temp_file} &> /dev/null
