/*****************************************************
 *
 * Gaussian elimination
 *
 * parallel version
 *
 *****************************************************/

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_SIZE 4096
#define MAX_WORKERS 128

typedef double matrix[MAX_SIZE][MAX_SIZE];

int	N;		/* matrix size		*/
int numWorkers; /* number of workers ( = threads ) */
int	maxnum;		/* max number of element*/
char	*Init;		/* matrix init type	*/
int	PRINT;		/* print switch		*/
matrix	A;		/* matrix A		*/
double	b[MAX_SIZE];	/* vector b             */
double	y[MAX_SIZE];	/* vector y             */
static uint8_t rowFinished[MAX_SIZE];  /* Array containing binary flag if each row is finished */

/* forward declarations */
void work(void);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
void Process_Row(int row);
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
	#pragma omp parallel for
    for(int i = 0; i<N; i++)
		Process_Row(i);
}

/* Elimination step: Using the result of the Division step for row *inputRow*,
 * modify row *outputRow*
 */
void Eliminate(int inputRow, int outputRow)
{
    for (int j = inputRow+1; j < N; j++)
    {
        A[outputRow][j] = A[outputRow][j] - A[outputRow][inputRow]*A[inputRow][j];
    }
    b[outputRow] = b[outputRow] - A[outputRow][inputRow]*y[inputRow];
    A[outputRow][inputRow] = 0.0;
}

/* Division step: Requires all rows before *row* to have undergone division and elimination steps */
void Division(int row)
{
    for (int j = row+1; j < N; j++)
    {
        A[row][j] = A[row][j] / A[row][row];
    }
    y[row] = b[row] / A[row][row];
    A[row][row] = 1.0;
}

/* Process a single row - this will read-access rows before this one and write-access only this row */
void Process_Row(int row)
{
    // Processing of a row depends on the successful processing of all previous rows
    for (int i=0; i<row; i++)
    {
        // Wait until the i-th row is processed
        while (!rowFinished[i]){}

        // Elimination step using the i-th row
        Eliminate(i, row);
    }

    // Division step
    Division(row);

    // Mark this row as finished
	//#pragma omp atomic
    rowFinished[row] = 1;
    // Wake up waiting threads threads as there is a new row finished
}

void Init_Matrix()
{
    int i, j;

    printf("\nsize      = %dx%d ", N, N);
    printf("\nmaxnum    = %d \n", maxnum);
    printf("Init	  = %s \n", Init);
    printf("Initializing matrix...");

    if (strcmp(Init,"rand") == 0) {
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++) {
        if (i == j) /* diagonal dominance */
            A[i][j] = (double)(rand() % maxnum) + 5.0;
        else
            A[i][j] = (double)(rand() % maxnum) + 1.0;
        }
    }
    }
    if (strcmp(Init,"fast") == 0) {
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
        if (i == j) /* diagonal dominance */
            A[i][j] = 5.0;
        else
            A[i][j] = 2.0;
        }
    }
    }

    /* Initialize vectors b and y */
    for (i = 0; i < N; i++) {
    b[i] = 2.0;
    y[i] = 1.0;
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
        printf("           [-w workers] worker count \n");
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
        case 'w':
        --argc;
        numWorkers = atoi(*++argv);
        break;
	    default:
		printf("%s: ignored option: -%s\n", prog, *argv);
		printf("HELP: try %s -u \n\n", prog);
		break;
	    } 
}
