#!/usr/bin/env bash

current_folder="${BASH_SOURCE%/*}"
results_file="${current_folder}/results/grid_search.csv"
temp_file="${current_folder}/.temp"

mkdir -p "${current_folder}/results" && touch ${results_file}

python "${current_folder}/../main.py" -a grid -p '{"cv": 3}' > ${temp_file}

score=`grep -oP "(?<=Best score: ).+" ${temp_file}`
time=`grep -oP "(?<=Took: )[\\d\\.]+" ${temp_file}`

echo "Time,Score" > ${results_file}
echo "${time},${score}" >> ${results_file}

rm -f ${temp_file} &> /dev/null
