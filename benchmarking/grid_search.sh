#!/usr/bin/env bash

current_folder="${BASH_SOURCE%/*}"
results_file="${current_folder}/results/grid_search.csv"
temp_file="${current_folder}/.temp"

mkdir -p "${current_folder}/results" && touch ${results_file}

echo "Max_Samples,Time,Score,Configs_Evaluated" > ${results_file}

for i in 1 2 3 4 5; do
    echo ${i}
    python -u "${current_folder}/../main.py" -a grid -p "{\"n_workers\": -1, \"max_samples\": ${i}, \"cv\": 3}" | tee ${temp_file}

    score=`grep -oP "(?<=Best score: ).+" ${temp_file}`
    time=`grep -oP "(?<=Took: )[\\d\\.]+" ${temp_file}`
    configs=`grep -oP "\\d+(?= candidates)" ${temp_file}`

    echo "${i},${time},${score},${configs}" >> ${results_file}
done

rm -f ${temp_file} &> /dev/null
