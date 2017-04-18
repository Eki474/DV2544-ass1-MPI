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
 * mpicc ./matmulMPI.c -o [Name of executable]
 * Run with:
 * mpirun -n [Number of cores(This case should be tested with: 1,2,4,8)] ./[Name of executable] 
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
static double d[SIZE][SIZE];

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
            c[i][j]=0;
            d[i][j]=0;
        }
}

static void
transp_matrix(void) {
  int temp;
  for(int i=0;i<SIZE; i++)
    for (int j=0; j<i;j++){
      temp = b[i][j];
      b[i][j] = b[j][i];
      b[j][i] = temp;
    }
}

 static void
 multi_offset(int offset_row, int rowSize, int offset_col, int colSize){
 		 for(int i=offset_row;i<offset_row+rowSize;i++){
  			  for(int j=offset_col;j<offset_col+colSize;j++){
            c[i][j]=0.0;
  				  for(int k=0;k<SIZE;k++){
  					  c[i][j]= c[i][j]+a[i][k]*b[j][k];
  				  }
  			  }
 		 }
}

static void
editResult(void)
{
	for(int i=0;i<SIZE;i++)
		for(int j=0;j<SIZE;j++)
			if(d[i][j]!=0.0) c[i][j] = d[i][j];
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
    printf("\n\nEND\n");
}

int
main(int argc, char **argv)
{
    int myrank, nproc, nprocT;
    int rows_node,cols_node; /* amount of work per node (rows per worker) */
    int mtype; /* message type: send/recv between master and workers */
    int dest, src, offset_rows,offset_cols;
    double start_time, end_time;
    int i, j, k;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    /* MASTER TASK */

    if (myrank == 0) {
        /* Initialization */
        printf("SIZE = %d, number of nodes = %d\n", SIZE, nproc);
        init_matrix();
        start_time = MPI_Wtime();
        transp_matrix();
        /* Send part of matrix a and the whole matrix b to workers */
        nprocT=nproc;
    	  rows_node = SIZE;
    	  cols_node = SIZE;

        while(nprocT>1){
        	if(rows_node>cols_node)
        		rows_node/=2;
        	else
        		cols_node/=2;
        nprocT/=2;
        }

        if(DEBUG) printf("\n\nPASS Before Master\n\n");
        mtype = FROM_MASTER;
        offset_rows = 0;
        offset_cols = 0;
        if (DEBUG) printf("R %d C %d", offset_rows,offset_cols);
        for (dest = 1; dest < nproc; dest++) {
            if (DEBUG)
            {
                printf("   Sending \noffset_rows : %d, offset_cols : %d , rows : %d, cols %d to task %d\n",offset_rows,offset_cols,rows_node,cols_node,dest);
                printf("   a[offset_rows][0] : %f, b[offset_cols][0] : %f\n",a[offset_rows][0],b[offset_cols][0]);
            }
            if(offset_rows<SIZE-rows_node)  offset_rows+=rows_node;
            else {
            	offset_rows=0;
            	offset_cols+=cols_node;
            }
            if(DEBUG)
            {
            printf("Offset Rows %d\n", offset_rows);
            printf("Offset Cols %d\n", offset_cols);
            }
            MPI_Send(&offset_rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&offset_cols, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows_node, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&cols_node, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&a[offset_rows][0], rows_node*SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&b[offset_cols][0], cols_node*SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
        }

        multi_offset(0,rows_node,0,cols_node);
        if(DEBUG) printf("\n\nPASS Before Worker\n\n");

        /* Collect the results from all the workers */
        mtype = FROM_WORKER;
        for (src = 1; src < nproc; src++) {
            /* Receive info from worker*/
            MPI_Recv(&offset_rows, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&offset_cols, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows_node, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&cols_node, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(d, SIZE*SIZE, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD, &status);
            editResult();
            if (DEBUG)
                printf("   Recvd %d rows %d cols from task %d, offset_rows = %d, offset_cols = %d\n", rows_node, cols_node, src, offset_rows, offset_cols);
        }
        end_time = MPI_Wtime();
        if (DEBUG)  print_matrix();/* Prints the resulting matrix c */
        printf("Execution time on %2d nodes: %f\n", nproc, end_time-start_time);

    } else {    /* Worker tasks */
        /* Receive data from master */
        if(DEBUG)
        {
            printf("WORKER TASK\nReceiving DATA from MASTER\n");
        }

        mtype = FROM_MASTER;
        MPI_Recv(&offset_rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&offset_cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows_node, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&cols_node, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a[offset_rows][0], rows_node*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b[offset_cols][0], cols_node*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);

        if (DEBUG)
        {
            printf ("Rank=%d, offsetRow=%d, row =%d, a[offsetRow][0]=%f \n",
                myrank, offset_rows, rows_node, a[offset_rows][0]);
            printf ("Rank=%d, offsetCols=%d, col =%d, b[0][offsetCol]=%f \n",
                myrank, offset_cols, cols_node, b[offset_cols][0]);
        }

        multi_offset(offset_rows,rows_node,offset_cols,cols_node);

        if (DEBUG)
        {
            printf("WORKER TASK AFTER MULTIPLICATION\n");
            printf("Rank=%d, offset_rows=%d, rows_node =%d, offset_cols=%d, cols_node=%d, c[offset][offset]=%f\n",
                myrank, offset_rows, rows_node, offset_cols, cols_node, c[offset_rows][offset_cols]);
        }
        /* Send the results to the master */
        mtype = FROM_WORKER;
        MPI_Send(&offset_rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
        MPI_Send(&offset_cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows_node, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
        MPI_Send(&cols_node, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
        MPI_Send(c,SIZE*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
        if(DEBUG)
        {
          printf("\n\nEND OF WORKER TASK\n\n");
        }
    }
    MPI_Finalize();
    return 0;
}
