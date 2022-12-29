#!/bin/bash

search_dir=/e/data/thessis_movies/analysis_B/
windows_sizes=(3  5 7 9 11 13)
thresholds=(1.5 1.8 2 2.3 2.5 2.7 3)
epsilons= (3 5 7 9)

# run this section for ARIMA and embedding vector option
for name in "$search_dir"/*; do
    for w in "${windows_sizes[@]}";  do
      for a in "${thresholds[@]}"; do
#        for e in "${epsilons[@]}"; do
            python3 analyze_univariant_embedding.py ${name} -o /e/data/results_1.12 -w ${w} -a ${a} -e 3 -s True -p False
            python3 analyze_univariant_embedding.py ${name} -o /e/data/results_1.12 -w ${w} -a ${a} -e 5 -s True -p False
#        done
      done
    done
  done

windows_sizes=(7)
# run this sectionfor deep_fcpred vector option
for name in "$search_dir"/*; do
    for w in "${windows_sizes[@]}";  do
       python3 analyze_univariant_embedding.py ${name} -o /e/data/results_1.12 -w ${w} -a -1 -e -1 -s True -p True
    done
done
