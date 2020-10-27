#!/bin/sh

for num in 1 5 10
do
    echo 'now is number '${num}
    path='../data/exp3/abis/'
    output='../data/exp3/top'${num}
    for file in `ls ${path}`
    do
        echo $file
        python3 SmartGift.py ${path}${file} ${output} ${num}
        echo '\n'
    done
done    
