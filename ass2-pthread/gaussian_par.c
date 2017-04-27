/*****************************************************
 *
 * Gaussian elimination
 *
 * parallel version
 *
 *****************************************************/

#include <stdio.h>
#include <pthread.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SIZE 4096
#define MAX_WORKERS 128

typedef double matrix[MAX_SIZE][MAX_SIZE];

typedef enum
{
    INIT_RAND,
    INIT_FAST
} MatrixInitMethod;

typedef struct
{
    int offset;
    int rows;
    MatrixInitMethod method;
} MatrixInitParams;

int	N;		/* matrix size		*/
int numWorkers; /* number of workers ( = threads ) */
int	maxnum;		/* max number of element*/
char	*Init;		/* matrix init type	*/
int	PRINT;		/* print switch		*/
matrix	A;		/* matrix A		*/
double	b[MAX_SIZE];	/* vector b             */
double	y[MAX_SIZE];	/* vector y             */

/* forward declarations */
void work(void);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
void Read_Options(int, char **);

int  main(int argc, char **argv)
{
    Init_Default();		/* Init default values	*/
    Read_Options(argc,argv);	/* Read arguments	*/
    Init_Matrix();		/* Init the matrix	*/
    work();
    if (PRINT == 1)
	Print_Matrix();
}

void work(void)
{
    int i, j, k;

    /* Gaussian elimination algorithm, Algo 8.4 from Grama */
    for (k = 0; k < N; k++) { /* Outer loop */
	for (j = k+1; j < N; j++)
	    A[k][j] = A[k][j] / A[k][k]; /* Division step */
	y[k] = b[k] / A[k][k];
	A[k][k] = 1.0;
	for (i = k+1; i < N; i++) {
	    for (j = k+1; j < N; j++)
		A[i][j] = A[i][j] - A[i][k]*A[k][j]; /* Elimination step */
	    b[i] = b[i] - A[i][k]*y[k];
	    A[i][k] = 0.0;
	}
    }
}

void *Init_Matrix_chunk(void* dataPtr)
{
    MatrixInitParams* params = (MatrixInitParams*)dataPtr;

    if(params->method == INIT_RAND)
    {
        for (int i = params->offset; i < params->offset + params->rows; i++){
            for (int j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i][j] = (double)(rand() % maxnum) + 5.0;
                else
                    A[i][j] = (double)(rand() % maxnum) + 1.0;
            }
        }
    }
    else if (params->method == INIT_FAST)
    {
        for (int i = params->offset; i < params->offset + params->rows; i++) {
            for (int j = 0; j < N; j++) {
            if (i == j) /* diagonal dominance */
                A[i][j] = 5.0;
            else
                A[i][j] = 2.0;
            }
        }
    }
    return NULL;
}

void Init_Matrix()
{
    printf("\nsize      = %dx%d ", N, N);
    printf("\nmaxnum    = %d \n", maxnum);
    printf("Init	  = %s \n", Init);
    printf("Initializing matrix...");
 
    assert(N % numWorkers == 0);
    int rowsPerWorker = N / numWorkers;
    int offset = rowsPerWorker;
    MatrixInitMethod initMethod;
    pthread_t threads[MAX_WORKERS];
    MatrixInitParams initParams[MAX_WORKERS];

    if (strcmp(Init,"rand") == 0)
        initMethod = INIT_RAND;
    else if (strcmp(Init,"fast") == 0)
        initMethod = INIT_FAST;
    else
        initMethod = INIT_RAND;

    // Spawn worker threads
    for (int worker = 1; worker < numWorkers; worker++)
    {
        initParams[worker].method = initMethod;
        initParams[worker].offset = offset;
        initParams[worker].rows = rowsPerWorker;

        pthread_create(&threads[worker], NULL, &Init_Matrix_chunk, &initParams[worker]);

        offset += rowsPerWorker;
    }

    // Do work on main thread
    initParams[0].method = initMethod;
    initParams[0].offset = 0;
    initParams[0].rows = rowsPerWorker;
    Init_Matrix_chunk(&initParams[0]);

    /* Initialize vectors b and y (main thread) */
    for (int i = 0; i < N; i++) {
    b[i] = 2.0;
    y[i] = 1.0;
    }

    // Join worker threads
    for (int worker = 1; worker < numWorkers; worker++)
    {
        pthread_join(threads[worker], NULL);
    }

    printf("done \n\n");
    if (PRINT == 1)
	Print_Matrix();
}

void Print_Matrix()
{
    int i, j;
 
    printf("Matrix A:\n");
    for (i = 0; i < N; i++) {
	printf("[");
	for (j = 0; j < N; j++)
	    printf(" %5.2f,", A[i][j]);
	printf("]\n");
    }
    printf("Vector b:\n[");
    for (j = 0; j < N; j++)
	printf(" %5.2f,", b[j]);
    printf("]\n");
    printf("Vector y:\n[");
    for (j = 0; j < N; j++)
	printf(" %5.2f,", y[j]);
    printf("]\n");
    printf("\n\n");
}

void Init_Default()
{
    N = 2048;
    numWorkers = 1;
    Init = "rand";
    maxnum = 15.0;
    PRINT = 0;
}
 
void Read_Options(int argc, char **argv)
{
    char    *prog;
 
    prog = *argv;
    while (++argv, --argc > 0)
	if (**argv == '-')
	    switch ( *++*argv ) {
	    case 'n':
		--argc;
		N = atoi(*++argv);
		break;
	    case 'h':
		printf("\nHELP: try sor -u \n\n");
		exit(0);
		break;
	    case 'u':
		printf("\nUsage: sor [-n problemsize]\n");
		printf("           [-D] show default values \n");
		printf("           [-h] help \n");
		printf("           [-I init_type] fast/rand \n");
		printf("           [-m maxnum] max random no \n");
		printf("           [-P print_switch] 0/1 \n");
		exit(0);
		break;
	    case 'D':
		printf("\nDefault:  n         = %d ", N);
		printf("\n          Init      = rand" );
		printf("\n          maxnum    = 5 ");
		printf("\n          P         = 0 \n\n");
		exit(0);
		break;
	    case 'I':
		--argc;
		Init = *++argv;
		break;
	    case 'm':
		--argc;
		maxnum = atoi(*++argv);
		break;
	    case 'P':
		--argc;
		PRINT = atoi(*++argv);
		break;
	    default:
		printf("%s: ignored option: -%s\n", prog, *argv);
		printf("HELP: try %s -u \n\n", prog);
		break;
	    } 
}
