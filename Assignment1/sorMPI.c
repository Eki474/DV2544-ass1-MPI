/*****************************************************
*             File : sorMPI.c
*        Author(s) : Rosen Sasov, Mohit Vellanki
*          Created : 2017-04-19
*    Last Modified : 2017-04-20
* Last Modified by : Mohit Vellanki
*
* Parallel version
*****************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <mpi.h>

#define MAX_SIZE 4096
#define EVEN_TURN 0 /* shall we calculate the 'red' or the 'black' elements */
#define ODD_TURN  1

#define FROM_MASTER 1	/* setting a message type */
#define FROM_WORKER 2	/* setting a message type */
#define DEBUG 0		/* 1 = debug on, 0 = debug off */

MPI_Status status;
typedef double matrix[MAX_SIZE+2][MAX_SIZE+2]; /* (+2) - boundary elements */

volatile struct globmem {
  int		N;		/* matrix size		*/
  int		maxnum;		/* max number of element*/
  char	*Init;		/* matrix init type	*/
  double	difflimit;	/* stop condition	*/
  double	w;		/* relaxation factor	*/
  int		PRINT;		/* print switch		*/
  matrix	A;		/* matrix A		*/
} *glob;

/* forward declarations */
int work(int rank, int nproc, int rows_node);
int Init_Matrix(int rank, int nproc);
void Print_Matrix();
void Init_Default();
int Read_Options(int, char **);

int
main(int argc, char **argv)
{
  int i, timestart, timeend, iter;
  int rankProc,rank,nproc,nprocT;
  int rows_node;
  glob = (struct globmem *) malloc(sizeof(struct globmem));

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankProc);
  nprocT = nproc;
  rank = rankProc;
  Init_Default();		/* Init default values	*/
  Read_Options(argc,argv);	/* Read arguments	*/
  rows_node = Init_Matrix(rank,nprocT);		/* Init the matrix	*/

  timestart= MPI_Wtime();
  iter = work(rank,nprocT,rows_node);

  if (glob->PRINT == 1) Print_Matrix();
  printf("\nNumber of iterations = %d\n", iter);

  MPI_Finalize();
  return 0;
}

int
work(int rank, int nproc,int rows_node)
{
  double prevmax_even, prevmax_odd, maxi, sum, w;
  int	m, n, N, i;
  int finished = 0;
  int turn = EVEN_TURN;
  int iteration = 0;
  int mtype,src,dest;
  int offset_rows;

  prevmax_even = 0.0;
  prevmax_odd = 0.0;
  N = glob->N;
  w = glob->w;
  offset_rows=0;

  while (!finished)
  {
    iteration++;

    //MASTER TASK
    if(rank==0)
    {
      offset_rows=1;
      //MASTER RECEIVING DATA FROM WORKERS AFTER FIRST ITERATION
      if(iteration>1)
      {
        mtype = FROM_WORKER;
        for(src=1;src<nproc;src++)
        {
          MPI_Recv(&offset_rows, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
          MPI_Recv(&rows_node, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
          MPI_Recv(&glob->A[offset_rows-1][1], rows_node*N, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD, &status);
          MPI_Recv(&glob->A[offset_rows][1], rows_node*N, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD, &status);
          MPI_Recv(&glob->A[offset_rows+1][1], rows_node*N, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD, &status);
        }
      }
    }
    //WORKER TASK
    else
    {
      if(offset_rows< N-offset_rows) ? offset_rows+=rows_node : offset_rows=1;
      if(iteration>1)
      {
        //WORKERS RECEIVING DATA FROM MASTER FOR THEIR WORK
        mtype = FROM_MASTER;
        MPI_Recv(&offset_rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows_node, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&glob->A[offset_rows-1][1], rows_node*N, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&glob->A[offset_rows][0], rows_node*N, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&glob->A[offset_rows+1][1], rows_node*N, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
      }
    }

    if (turn == EVEN_TURN) {
      /* CALCULATE part A - even elements */
      for(i=0;i<rows_node;i++)
      {
        for (m = i+offset_rows; m < N+1; m++) {
          for (n = 1; n < N+1; n++) {
            if (((m + n) % 2) == 0)
            {
              glob->A[m][n] = (1 - w) * glob->A[m][n]
              + w * (glob->A[m-1][n] + glob->A[m+1][n]
                + glob->A[m][n-1] + glob->A[m][n+1]) / 4;
              }
            }
          }
        }

        /* Calculate the maximum sum of the elements */
        maxi = -999999.0;
        for (m = 1; m < N+1; m++) {
          sum = 0.0;
          for (n = 1; n < N+1; n++)
          sum += glob->A[m][n];
          if (sum > maxi)
          maxi = sum;
        }

        /* Compare the sum with the prev sum, i.e., check wether
        * we are finished or not. */
        if (fabs(maxi - prevmax_even) <= glob->difflimit)
        finished = 1;
        if ((iteration%100) == 0)
        printf("Iteration: %d, maxi = %f, prevmax_even = %f\n",
        iteration, maxi, prevmax_even);
        prevmax_even = maxi;
        turn = ODD_TURN;
      }

      else if (turn == ODD_TURN)
      {
        /* CALCULATE part B - odd elements*/
        for(i=0;i<rows_node;i++)
        {
          for (m = i+offset_rows; m < N+1; m++) {
            for (n = 1; n < N+1; n++) {
              if (((m + n) % 2) == 1)
              glob->A[m][n] = (1 - w) * glob->A[m][n]
              + w * (glob->A[m-1][n] + glob->A[m+1][n]
                + glob->A[m][n-1] + glob->A[m][n+1]) / 4;
              }
            }
          }
          /* Calculate the maximum sum of the elements */
          maxi = -999999.0;
          for (m = 1; m < N+1; m++) {
            sum = 0.0;
            for (n = 1; n < N+1; n++)
            sum += glob->A[m][n];
            if (sum > maxi)
            maxi = sum;
          }
          /* Compare the sum with the prev sum, i.e., check wether
          * we are finished or not. */
          if (fabs(maxi - prevmax_odd) <= glob->difflimit)
          finished = 1;
          if ((iteration%100) == 0)
          printf("Iteration: %d, maxi = %f, prevmax_odd = %f\n",
          iteration, maxi, prevmax_odd);
          prevmax_odd = maxi;
          turn = EVEN_TURN;
        }
        else
        {
          /* something is very wrong... */
          printf("PANIC: Something is really wrong!!!\n");
          exit(-1);
        }

        if (iteration > 100000)
        {
          /* exit if we don't converge fast enough */
          printf("Max number of iterations reached! Exit!\n");
          finished = 1;
        }
        // MASTER TASK
        if(rank==0 && !finished)
        {
          mtype = FROM_MASTER;

          //MASTER SENDING TO WORKERS

          for (dest = 1; dest < nproc; dest++) {
            /* code */
           if (offset_rows<N-offset_rows)  offset_rows+=rows_node;
           else offset_rows=1;

            MPI_Send(&offset_rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows_node,1,MPI_INT,dest,mtype, MPI_COMM_WORLD);
            MPI_Send(&glob->A[offset_rows-1][1], N, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&glob->A[offset_rows][0], N+2, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&glob->A[offset_rows+1][1], N, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
          }
        }

        // WORKERS TASK
        else
        {
          // SENDING DATA FROM WORKER TO MASTER
          mtype = FROM_WORKER;
          MPI_Send(&offset_rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
          MPI_Send(&rows_node,1,MPI_INT,0,mtype, MPI_COMM_WORLD);
          MPI_Send(&glob->A[offset_rows-1][1], N, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
          MPI_Send(&glob->A[offset_rows][0], rows_node*(N+2), MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
          MPI_Send(&glob->A[offset_rows+1][1], N, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
        }

      } //end while
      return iteration;
    }

    /*--------------------------------------------------------------*/

    int
    Init_Matrix(int rank, int nproc)
    {
      int i, j, N, dmmy;
      int mtype,dest,src;
      int offset_rows, nprocT, rows_node;
      N = glob->N;
      printf("\nsize      = %dx%d ",N,N);
      printf("\nmaxnum    = %d \n",glob->maxnum);
      printf("difflimit = %.7lf \n",glob->difflimit);
      printf("Init	  = %s \n",glob->Init);
      printf("w	  = %f \n\n",glob->w);
      printf("Initializing matrix...");

      /* Initialize all grid elements, including the boundary */
      for (i = 0; i < glob->N+2; i++) {
        for (j = 0; j < glob->N+2; j++) {
          glob->A[i][j] = 0.0;
        }
      }

      if (strcmp(glob->Init,"count") == 0) {
        for (i = 1; i < N+1; i++){
          for (j = 1; j < N+1; j++) {
            glob->A[i][j] = (double)i/2;
          }
        }
      }

      if (strcmp(glob->Init,"rand") == 0) {
        for (i = 1; i < N+1; i++){
          for (j = 1; j < N+1; j++) {
            glob->A[i][j] = (rand() % glob->maxnum) + 1.0;
          }
        }
      }

      if (strcmp(glob->Init,"fast") == 0) {
        for (i = 1; i < N+1; i++){
          dmmy++;
          for (j = 1; j < N+1; j++) {
            dmmy++;
            if ((dmmy%2) == 0)  glob->A[i][j] = 1.0;
            else  glob->A[i][j] = 5.0;
          }
        }
      }

      /* Set the border to the same values as the outermost rows/columns */
      /* fix the corners */
      glob->A[0][0] = glob->A[1][1];
      glob->A[0][N+1] = glob->A[1][N];
      glob->A[N+1][0] = glob->A[N][1];
      glob->A[N+1][N+1] = glob->A[N][N];
      /* fix the top and bottom rows */
      for (i = 1; i < N+1; i++) {
        glob->A[0][i] = glob->A[1][i];
        glob->A[N+1][i] = glob->A[N][i];
      }
      /* fix the left and right columns */
      for (i = 1; i < N+1; i++) {
        glob->A[i][0] = glob->A[i][1];
        glob->A[i][N+1] = glob->A[i][N];
      }

      printf("done \n\n");
      if (glob->PRINT == 1) Print_Matrix();

      if(rank==0)
      {
        nprocT=nproc;
        rows_node=N;

        while(nprocT>1)
        {
          rows_node/=2;
          nprocT/=2;
        }

        mtype = FROM_MASTER;
        offset_rows=1;

        for(dest=1;dest<nproc;dest++)
        {
          if(offset_rows< N-offset_rows) offset_rows+=rows_node;
          else offset_rows=1;

          MPI_Send(&rows_node,1,MPI_INT,dest,mtype, MPI_COMM_WORLD);
          MPI_Send(&offset_rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
          MPI_Send(&glob->A[offset_rows-1][1], N, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
          MPI_Send(&glob->A[offset_rows][0], N+2, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
          MPI_Send(&glob->A[offset_rows+1][1], N, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
        }
      }
      else
      {
        mtype = FROM_MASTER;
        MPI_Recv(&rows_node, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&offset_rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&glob->A[offset_rows-1][1], N, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&glob->A[offset_rows][0], N+2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&glob->A[offset_rows+1][1], N, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
      }

      return rows_node;

    }

    void
    Print_Matrix()
    {
      int i, j, N;

      N = glob->N;
      for (i=0; i<N+2 ;i++){
        for (j=0; j<N+2 ;j++){
          printf(" %f",glob->A[i][j]);
        }
        printf("\n");
      }
      printf("\n\n");
    }

    void
    Init_Default()
    {
      glob->N = 2048;
      glob->difflimit = 0.00001*glob->N;
      glob->Init = "rand";
      glob->maxnum = 15.0;
      glob->w = 0.5;
      glob->PRINT = 0;
    }

    int
    Read_Options(int argc, char **argv)
    {
      char    *prog;

      prog = *argv;
      while (++argv, --argc > 0)
      if (**argv == '-')
      switch ( *++*argv ) {
        case 'n':
        --argc;
        glob->N = atoi(*++argv);
        glob->difflimit = 0.00001*glob->N;
        break;
        case 'h':
        printf("\nHELP: try sor -u \n\n");
        exit(0);
        break;
        case 'u':
        printf("\nUsage: sor [-n problemsize]\n");
        printf("           [-d difflimit] 0.1-0.000001 \n");
        printf("           [-D] show default values \n");
        printf("           [-h] help \n");
        printf("           [-I init_type] fast/rand/count \n");
        printf("           [-m maxnum] max random no \n");
        printf("           [-P print_switch] 0/1 \n");
        printf("           [-w relaxation_factor] 1.0-0.1 \n\n");
        exit(0);
        break;
        case 'D':
        printf("\nDefault:  n         = %d ", glob->N);
        printf("\n          difflimit = 0.0001 ");
        printf("\n          Init      = rand" );
        printf("\n          maxnum    = 5 ");
        printf("\n          w         = 0.5 \n");
        printf("\n          P         = 0 \n\n");
        exit(0);
        break;
        case 'I':
        --argc;
        glob->Init = *++argv;
        break;
        case 'm':
        --argc;
        glob->maxnum = atoi(*++argv);
        break;
        case 'd':
        --argc;
        glob->difflimit = atof(*++argv);
        break;
        case 'w':
        --argc;
        glob->w = atof(*++argv);
        break;
        case 'P':
        --argc;
        glob->PRINT = atoi(*++argv);
        break;
        default:
        printf("%s: ignored option: -%s\n", prog, *argv);
        printf("HELP: try %s -u \n\n", prog);
        break;
      }
    }
