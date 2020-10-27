#!/bin/sh

path='../data/exp2/vulnerableContracts/'
output='../data/exp2/test_seeds/'
for directory in `ls ${path}`
do
    for file in `cd ${path}${directory}'/'&&ls`
    do
        echo ${path}${directory}/${file}
        echo '\n'
        python3 using_ML.py ${path}${directory}/${file} ${output}${directory}
        echo '\n'
    done
done
    