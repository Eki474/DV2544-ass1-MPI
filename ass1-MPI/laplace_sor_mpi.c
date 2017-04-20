/*****************************************************
 *
 * S O R algorithm
 * ("Red-Black" solution to LaPlace approximation)
 *
 * sequential version
 *
 *****************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <float.h>
#include <mpi.h>
#include <assert.h>

#define MAX_SIZE 4096
#define EVEN_TURN 0 /* shall we calculate the 'red' or the 'black' elements */
#define ODD_TURN  1

/* message types */
#define FROM_MASTER 1
#define FROM_WORKER 2
#define FINISHEDFLAG_FROM_WORKER 3
#define FINISHEDFLAG_FROM_MASTER 4
#define WORKER_TO_WORKER 5

MPI_Status status;

typedef double **matrix;

volatile struct globmem {
    int N;        /* matrix size		*/
    int maxnum;        /* max number of element*/
    char *Init;        /* matrix init type	*/
    double difflimit;    /* stop condition	*/
    double w;        /* relaxation factor	*/
    int PRINT;        /* print switch		*/
    matrix A;        /* matrix A		*/
    int nproc;
    int myrank;
} *glob;

/* forward declarations */
int work(int n_rows, int offset);

void Init_Matrix();

void Print_Matrix();

void Init_Default();

void Read_Options(int, char **);

int
main(int argc, char **argv) {
    int timestart, timeend, iter;

    glob = (struct globmem *) malloc(sizeof(struct globmem));

    MPI_Init(&argc, &argv);

    printf("SIZE = %d, number of nodes = %d\n", glob->N, glob->nproc);
    Init_Default();        /* Init default values	*/
    Read_Options(argc, argv);    /* Read arguments	*/

    if(glob->myrank == 0) {
        Init_Matrix();        /* Init the matrix	*/
        timestart = MPI_Wtime();

        //split the matrix
        assert(glob->nproc > 0);
        assert(glob->N % glob->nproc == 0);
        int rows = glob->N / glob->nproc;
        //send to workers
        int offset = rows + 1; // First is extra
        for(int i = 1; i < glob->nproc; i++){
            MPI_Send(&offset, 1, MPI_INT, i, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, i, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&glob->A[offset-1][0], (rows + 2) * (glob->N + 2), MPI_DOUBLE, i, FROM_MASTER, MPI_COMM_WORLD);
            offset += rows;
        }
        //work work work
        iter = work(rows, 1);
        //receive results
        for(int i = 1; i < glob->nproc; i++){
            MPI_Recv(&offset, 1, MPI_INT, i, FROM_WORKER, MPI_COMM_WORLD, &status);
            printf("Master will receive %d rows from offset %d from worker %d\n", rows, offset, i);
            MPI_Recv(&glob->A[offset][0], rows * (glob->N + 2), MPI_DOUBLE, i, FROM_WORKER, MPI_COMM_WORLD, &status);
        }
        timeend = MPI_Wtime();
        if (glob->PRINT == 1)
            Print_Matrix();
        printf("\nNumber of iterations = %d\n", iter);
    }else {
        int nb_rows, offset;
        MPI_Recv(&offset, 1, MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&nb_rows, 1, MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&glob->A[offset-1][0], (nb_rows + 2) * (glob->N + 2), MPI_DOUBLE, 0, FROM_MASTER, MPI_COMM_WORLD, &status);
        //workers job
        printf("Worker %d received %d rows\n", glob->myrank, nb_rows);
        printf("Worker %d received %d offset\n", glob->myrank, offset);
        iter = work(nb_rows, offset);
        //send back result to master
        MPI_Send(&offset, 1, MPI_INT, 0, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&glob->A[offset][0], nb_rows * (glob->N + 2), MPI_DOUBLE, 0, FROM_WORKER, MPI_COMM_WORLD);
    }


    MPI_Finalize();
    return 0;
}

/* Execute one turn of the SOR algorithm on a chunk of data */
void laplace_sor(int n_rows, int offset, int isOddTurn)
{
    int n_cols = glob->N;
    double w = glob->w;
    for (int m = offset; m < offset + n_rows; m++) {
        for (int n = 1; n < n_cols + 1; n++) {
            if (((m + n) % 2) == isOddTurn) {
                glob->A[m][n] = (1 - w) * glob->A[m][n]
                                + w * (glob->A[m - 1][n] + glob->A[m + 1][n]
                                       + glob->A[m][n - 1] + glob->A[m][n + 1]) / 4;
            }
        }
    }
}

int
work(int n_rows, int offset) {
    double maxi, sum;
    int m, n, n_cols;
    int finished = 0;
    int turn = EVEN_TURN;
    int iteration = 0;
    double prevmax[2] = { 0.0, 0.0 }; // Contains prevmax_even and prevmax_odd

    n_cols = glob->N;

    while (!finished) {
        iteration++;
        // Exchange rows between workers
        if(glob->myrank != 0) {
            //printf("%f - Worker %d will receive above-top row of worker %d\n", MPI_Wtime(), glob->myrank, glob->myrank - 1);
            MPI_Recv(&glob->A[offset-1][0], glob->N, MPI_DOUBLE, glob->myrank - 1, WORKER_TO_WORKER, MPI_COMM_WORLD, &status);
            //printf("%f - Worker %d will send top row to worker %d\n", MPI_Wtime(), glob->myrank, glob->myrank - 1);
            MPI_Send(&glob->A[offset][0], glob->N, MPI_DOUBLE, glob->myrank - 1, WORKER_TO_WORKER, MPI_COMM_WORLD);
        }
        if(glob->myrank != glob->nproc-1) {
            //printf("%f - Worker %d will send bottom row to worker %d\n", MPI_Wtime(), glob->myrank, glob->myrank + 1);
            MPI_Send(&glob->A[offset + n_rows][0], glob->N, MPI_DOUBLE, glob->myrank + 1, WORKER_TO_WORKER, MPI_COMM_WORLD);
            //printf("%f - Worker %d will receive below-bottom row from worker %d\n", MPI_Wtime(), glob->myrank, glob->myrank + 1);
            MPI_Recv(&glob->A[offset + n_rows + 1][0], glob->N, MPI_DOUBLE, glob->myrank + 1, WORKER_TO_WORKER, MPI_COMM_WORLD, &status);
        }

        /* CALCULATE */
        laplace_sor(n_rows, offset, turn);

        /* Calculate the maximum sum of the elements */
        maxi = DBL_MIN;
        for (m = offset; m < offset + n_rows; m++) {
            sum = 0.0;
            for (n = 1; n < n_cols + 1; n++)
                sum += glob->A[m][n];
            if (sum > maxi)
                maxi = sum;
        }
        /* Compare the sum with the prev sum, i.e., check wether
         * we are finished or not. */
        if (fabs(maxi - prevmax[turn]) <= glob->difflimit)
            finished = 1;
        if ((iteration % 100) == 0)
            printf("Iteration: %d, maxi = %f, prevmax[%d] = %f\n",
                   iteration, maxi, turn, prevmax[turn]);
        prevmax[turn] = maxi;

        turn = turn == EVEN_TURN ? ODD_TURN : EVEN_TURN;

        if (turn != EVEN_TURN && turn != ODD_TURN) {
            /* something is very wrong... */
            printf("PANIC: Something is really wrong!!!\n");
            exit(-1);
        }
        if (iteration > 100000) {
            /* exit if we don't converge fast enough */
            printf("Max number of iterations reached! Exit!\n");
            finished = 1;
        }
        if(glob->myrank != 0){
            /* Every worker (except for the master) sends their finished flag to the master */
            MPI_Send(&finished, 1, MPI_INT, 0, FINISHEDFLAG_FROM_WORKER, MPI_COMM_WORLD);
            /* Receive the finished flag (in case one worker is finished, all workers have to stop */
            MPI_Recv(&finished, 1, MPI_INT, 0, FINISHEDFLAG_FROM_MASTER, MPI_COMM_WORLD, &status);
        }else {
            /* Master stops all workers if at least one worker hit the terminating condition */
            int temp = 0;
            int p;
            for(p = 1; p < glob->nproc; p++){
                MPI_Recv(&temp, 1, MPI_INT, p, FINISHEDFLAG_FROM_WORKER, MPI_COMM_WORLD, &status);
                finished = finished || temp;
            }
            for(p = 1; p < glob->nproc; p++){
                MPI_Send(&finished, 1, MPI_INT, p, FINISHEDFLAG_FROM_MASTER, MPI_COMM_WORLD);
            }
        }
    }
    return iteration;
}

/*--------------------------------------------------------------*/

void
Init_Matrix() {
    int i, j, N;

    N = glob->N;
    printf("\nsize      = %dx%d ", N, N);
    printf("\nmaxnum    = %d \n", glob->maxnum);
    printf("difflimit = %.7lf \n", glob->difflimit);
    printf("Init	  = %s \n", glob->Init);
    printf("w	  = %f \n\n", glob->w);
    printf("Initializing matrix...");

    /* Initialize all grid elements, including the boundary */
    for (i = 0; i < glob->N + 2; i++) {
        for (j = 0; j < glob->N + 2; j++) {
            glob->A[i][j] = 0.0;
        }
    }
    if (strcmp(glob->Init, "count") == 0) {
        for (i = 1; i < N + 1; i++) {
            for (j = 1; j < N + 1; j++) {
                glob->A[i][j] = (double) i / 2;
            }
        }
    }
    if (strcmp(glob->Init, "rand") == 0) {
        for (i = 1; i < N + 1; i++) {
            for (j = 1; j < N + 1; j++) {
                glob->A[i][j] = (rand() % glob->maxnum) + 1.0;
            }
        }
    }
    if (strcmp(glob->Init, "fast") == 0) {
        int dmmy = 0;
        for (i = 1; i < N + 1; i++) {
            dmmy++;
            for (j = 1; j < N + 1; j++) {
                dmmy++;
                if ((dmmy % 2) == 0)
                    glob->A[i][j] = 1.0;
                else
                    glob->A[i][j] = 5.0;
            }
        }
    }

    /* Set the border to the same values as the outermost rows/columns */
    /* fix the corners */
    glob->A[0][0] = glob->A[1][1];
    glob->A[0][N + 1] = glob->A[1][N];
    glob->A[N + 1][0] = glob->A[N][1];
    glob->A[N + 1][N + 1] = glob->A[N][N];
    /* fix the top and bottom rows */
    for (i = 1; i < N + 1; i++) {
        glob->A[0][i] = glob->A[1][i];
        glob->A[N + 1][i] = glob->A[N][i];
    }
    /* fix the left and right columns */
    for (i = 1; i < N + 1; i++) {
        glob->A[i][0] = glob->A[i][1];
        glob->A[i][N + 1] = glob->A[i][N];
    }

    printf("done \n\n");
    if (glob->PRINT == 1)
        Print_Matrix();
}

void
Print_Matrix() {
    int i, j, N;

    N = glob->N;
    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < N + 2; j++) {
            printf(" %f", glob->A[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

void
Init_Default() {
    glob->N = 2048;
    glob->difflimit = 0.00001 * glob->N;
    glob->Init = "rand";
    glob->maxnum = 15.0;
    glob->w = 0.5;
    glob->PRINT = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &(glob->nproc));
    MPI_Comm_rank(MPI_COMM_WORLD, &(glob->myrank));

    // Dynamically allocate contiguous 2D-array
    // https://stackoverflow.com/a/29375830
    int sidelength = glob->N + 2;
    glob->A = malloc(sidelength * sizeof(double));
    glob->A[0] = malloc(sidelength * sidelength * sizeof(double));
    for (int i = 1; i < sidelength; i++)
        glob->A[i] = glob->A[i-1] + sidelength;
}

void
Read_Options(int argc, char **argv) {
    char *prog;

    prog = *argv;
    while (++argv, --argc > 0)
        if (**argv == '-')
            switch (*++*argv) {
                case 'n':
                    --argc;
                    glob->N = atoi(*++argv);
                    glob->difflimit = 0.00001 * glob->N;
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
                    printf("\n          Init      = rand");
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
