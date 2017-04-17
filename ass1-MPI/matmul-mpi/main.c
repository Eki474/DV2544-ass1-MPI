/***************************************************************************
 *
 * MPI-version of block-wise Matrix-Matrix multiplication
 *
 *             File : main.c
 *        Author(s) : Lucie Labadie
 *          Created : 2017-04-06
 *    Last Modified : 2017-04-17
 * Last Modified by : Lucie Labadie
 *
 ***************************************************************************/


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

/*
 * transpose the b matrix from rwo base to column base (working)
 */
static void transpose_matrix(void)
{
    double swap;
    int i, j;
    for(i=1; i < SIZE; i++)
        for(j=0; j < SIZE; j++)
            if(j < i)
            {
                swap = b[SIZE-i][SIZE-j];
                b[SIZE-i][SIZE-j] = b[i][j];
                b[i][j] = swap;
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

    if (myrank == 0)//Master
    {
        /* Initialization */
        printf("SIZE = %d, number of nodes = %d\n", SIZE, nproc);
        init_matrix();
        transpose_matrix();
        start_time = MPI_Wtime();
    }else //workers
    {
        //
    }
    //To be continued...

    return 0;
}
