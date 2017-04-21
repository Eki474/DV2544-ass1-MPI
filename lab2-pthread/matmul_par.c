/***************************************************************************
 *
 * Sequential version of Matrix-Matrix multiplication
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define SIZE 1024

static double a[SIZE][SIZE];
static double b[SIZE][SIZE];
static double c[SIZE][SIZE];

static pthread_t threads[SIZE];

typedef struct
{
    int num_rows;
    int offset;
} MatmulParamT;

static void
init_matrix(void)
{
    int i, j;

    for (i = 0; i < SIZE; i++)
        for (j = 0; j < SIZE; j++) {
	    /* Simple initialization, which enables us to easy check
	     * the correct answer. Each element in c will have the same 
	     * value as SIZE after the matmul operation.
	     */
	    a[i][j] = 1.0;
	    b[i][j] = 1.0;
        }
}

static void matmul(int num_rows, int offset)
{
    for (int i = offset; i < offset + num_rows; i++) {
        for (int j = 0; j < SIZE; j++) {
            c[i][j] = 0.0;
            for (int k = 0; k < SIZE; k++)
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
        }
    }
}

static void* matmul_wrapper(void* dataPtr)
{
    MatmulParamT* params = (MatmulParamT*)dataPtr;
    matmul(params->num_rows, params->offset);
    free(params);
    return 0;
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
    init_matrix();

    // Start threads
    for (int i=0; i<SIZE; i++)
    {
        MatmulParamT* params = malloc(sizeof(MatmulParamT));
        params->num_rows = 1;
        params->num_rows = i;

        if(pthread_create(&threads[i], NULL, &matmul_wrapper, params))
        {
            printf("Error creating thread %d\n", i);
            return 1;
        }
    }

    // Join threads
    for (int i=0; i<SIZE; i++)
    {
        if(pthread_join(threads[i], NULL))
        {
            printf("Error joining thread %d\n", i);
            return 1;
        }
    }

    //matmul_seq();
    //print_matrix();
}

