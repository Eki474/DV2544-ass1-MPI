#!/bin/bash

nodes=(1 2 4 8)

echo "" > measurement_gauss.txt

echo "sequential" >> measurement_gauss.txt
for i in `seq 1 1 10`
do
	(time ./gaussian_seq > /dev/null) 2>&1 | grep real >> measurement_gauss.txt
done

for n in "${nodes[@]}"
do
	echo "$n nodes" >> measurement_gauss.txt

	for i in `seq 1 1 10`
	do
		(time ./gaussian_par -w $n > /dev/null) 2>&1 | grep real >> measurement_gauss.txt	
	done
done

