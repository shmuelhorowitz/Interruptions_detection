#!/bin/bash

search_dir=/e/data/video/
for entry in "$search_dir"/*
do
  echo "$entry"
  python3 main_analize_B.py "$entry" /e/Dev/Thessis/config.yaml
done

