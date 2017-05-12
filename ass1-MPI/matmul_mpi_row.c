/***************************************************************************
 *
 * MPI-version of row-wise Matrix-Matrix multiplication
 *
 *             File : matmul_mpi.c
 *        Author(s) : Håkan Grahn
 *          Created : 2009-01-30
 *    Last Modified : 2009-01-30
 * Last Modified by : Håkan Grahn
 *
 * (c) 2009-2017 by Håkan Grahn, Blekinge Institute of Technology.
 * All Rights Reserved
 ***************************************************************************/

/*
 * Compile with:
 * mpicc -o mm matmul_mpi.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE 1024	/* assumption: SIZE a multiple of number of nodes */
        /* SIZE should be 1024 in our measurements in the assignment */
        /* Hint: use small sizes when testing, e.g., SIZE 8 */
#define FROM_MASTER 1	/* setting a message type */
#define FROM_WORKER 2	/* setting a message type */
#define DEBUG 0		/* 1 = debug on, 0 = debug off */

MPI_Status status;

static double a[SIZE][SIZE];
static double b[SIZE][SIZE];
static double c[SIZE][SIZE];

static void
init_matrix(void)
{
    int i, j;

    for (i = 0; i < SIZE; i++)
        for (j = 0; j < SIZE; j++) {
            /* Simple initialization, which enables us to easy check
             * the correct answer. Given SIZE size of the matrices, then
             * the output should be
             *     SIZE ... 2*SIZE ...
             *     ...
             *     2*SIZE ... 4*SIZE ...
             *     ...
             */
            a[i][j] = 1.0;
            if (i >= SIZE/2) a[i][j] = 2.0;
            b[i][j] = 1.0;
            if (j >= SIZE/2) b[i][j] = 2.0;
        }
}

static void
print_matrix(void)
{
    int i, j;

    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++)
            printf(" %7.2f", c[i][j]);
        printf("\n");
    }
}

int
main(int argc, char **argv)
{
    int myrank, nproc;
    int rows; /* amount of work per node (rows per worker) */
    int mtype; /* message type: send/recv between master and workers */
    int dest, src, offset;
    double start_time, end_time;
    int i, j, k;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {    /* Master task */
        /* Initialization */
        printf("SIZE = %d, number of nodes = %d\n", SIZE, nproc);
        init_matrix();
        start_time = MPI_Wtime();

        /* Send part of matrix a and the whole matrix b to workers */
        rows = SIZE / nproc;
        mtype = FROM_MASTER;
        offset = rows;
        for (dest = 1; dest < nproc; dest++) {
            if (DEBUG)
                printf("   sending %d rows to task %d\n",rows,dest);
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows*SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&b, SIZE*SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            offset += rows;
        }

        /* let master do its part of the work */
        for (i = 0; i < rows; i++) {
            for (j = 0; j < SIZE; j++) {
                c[i][j] = 0.0;
                for (k = 0; k < SIZE; k++)
                    c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }

        /* collect the results from all the workers */
        mtype = FROM_WORKER;
        for (src = 1; src < nproc; src++) {
            MPI_Recv(&offset, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows*SIZE, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD, &status);
            if (DEBUG)
                printf("   recvd %d rows from task %d, offset = %d\n", rows, src, offset);
        }

        end_time = MPI_Wtime();
        if (DEBUG)
            /* Prints the resulting matrix c */
            print_matrix();
        printf("Execution time on %2d nodes: %f\n", nproc, end_time-start_time);

    } else {    /* Worker tasks */
        /* Receive data from master */
        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a[offset][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, SIZE*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
        if (DEBUG)
            printf ("Rank=%d, offset=%d, row =%d, a[offset][0]=%f, b[0][0]=%f\n",
                myrank, offset, rows, a[offset][0], b[0][0]);

        /* do the workers part of the calculation */
        for (i=offset; i<offset+rows; i++)
            for (j=0; j<SIZE; j++) {
                c[i][j] = 0.0;
                for (k=0; k<SIZE; k++)
                    c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        if (DEBUG)
            printf("Rank=%d, offset=%d, row =%d, c[offset][0]=%e\n",
                myrank, offset, rows, a[offset][0]);

        /* send the results to the master */
        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
        MPI_Send(&c[offset][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
