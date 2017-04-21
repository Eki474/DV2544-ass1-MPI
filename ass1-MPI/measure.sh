#!/bin/bash

nodes=(1 2 4 8)

echo "" > measurement_matrix.txt
echo "" > measurement_laplace.txt

./matrix_mult_seq | grep Execution >> measurement_matrix.txt
./laplace_sor_seq | grep Execution >> measurement_laplace.txt

for n in "${nodes[@]}"
do
	for i in `seq 1 1 10`
	do
		mpirun -n $n ./matrix_mult_mpi2 | grep Execution >> measurement_matrix.txt
	done
	for i in `seq 1 1 10`
	do
		mpirun -n $n ./laplace_sor_mpi  | grep Execution >> measurement_laplace.txt
	done
done

#for i in `seq 1 1 10`; do mpirun -n 1 ./laplace_sor_mpi | grep Execution >> measurement_laplace_1.txt ; done
