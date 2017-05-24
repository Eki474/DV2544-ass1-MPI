#!/bin/bash

echo "" > measurement_qsort.txt
echo "" > measurement_gaussian.txt

echo "sequential" | tee -a measurement_qsort.txt | tee -a measurement_gaussian.txt
for i in `seq 1 1 10`
do
	echo $i
	(time ./qsort_seq > /dev/null) 2>&1 | grep real | tee -a measurement_qsort.txt
	(time ./gaussian_seq > /dev/null) 2>&1 | grep real | tee -a measurement_gaussian.txt
done

echo "OpenMP" | tee -a measurement_qsort.txt | tee -a measurement_gaussian.txt
for i in `seq 1 1 10`
do
	echo $i
	(time ./qsort_par > /dev/null) 2>&1 | grep real | tee -a measurement_qsort.txt	
	(time ./gaussian_par > /dev/null) 2>&1 | grep real | tee -a measurement_gaussian.txt	
done

