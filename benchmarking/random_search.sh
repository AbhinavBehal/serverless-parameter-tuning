#!/usr/bin/env bash

current_folder="${BASH_SOURCE%/*}"
results_file="${current_folder}/results/random_search.csv"
temp_file="${current_folder}/.temp"

mkdir -p "${current_folder}/results" && touch ${results_file}

echo "Iterations,Time,Score" > ${results_file}

for i in 1 2 4 8 16 32 64 128 256; do
    python "${current_folder}/../main.py" -a random -p "{\"n_iter\": ${i}, \"cv\": 3}" > ${temp_file}
    score=`grep -oP "(?<=Best score: ).+" ${temp_file}`
    time=`grep -oP "(?<=Took: )[\\d\\.]+" ${temp_file}`
    echo "${i},${time},${score}" >> ${results_file}
done

rm -f ${temp_file} &> /dev/null
