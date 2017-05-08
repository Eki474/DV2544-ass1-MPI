/*****************************************************
*
* Gaussian elimination
*
* Parallel version
*
* Author(s) : Rosen Sasov, Mohit Vellanki
*
*****************************************************/


/*  INFORMATION FOR COMPILING AND EXECUTING
*
* To Compile use :
* gcc gaussian_pthreads.c -o [name of executable] -pthread
* To Run use :
* [name of executable] -n [number of cores] -N [size of matrix] -P [Print mode : 0-off/ 1-on]
* Values by default : Check Init_Default function
*
*/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>


#define MAX_SIZE 4096

typedef double matrix[MAX_SIZE][MAX_SIZE];

int	N;		/* Matrix size */
int nproc; /* Number of processors */
int	maxnum;		/* Max number of element */
char	*Init;		/* Matrix init type	*/
int	PRINT;		/* Print switch */
matrix	A;		/* Matrix A  */
double	b[MAX_SIZE];	/* Vector b */
double	y[MAX_SIZE];	/* Vector y */
char *flagTab = NULL; /* Flag table */
int threadCnt; /* Global thread counter */
pthread_mutex_t lock; /*  Mutex */
pthread_cond_t cond = PTHREAD_COND_INITIALIZER; /*  Condition */


/* Forward declarations */
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
int Read_Options(int, char **);
void * work(void* id_T);

int
main(int argc, char **argv)
{
  int i;
  unsigned int *id_T = NULL;
  pthread_t *pThread = NULL;
  char *flags;
  threadCnt = 0;

  /* Init default values	*/
  Init_Default();

  /* Read arguments	*/
  Read_Options(argc,argv);

  /* Allocating memory */
  id_T = malloc(nproc*sizeof(unsigned int)); // Thread IDs
  pThread = malloc(nproc*sizeof(pthread_t)); // pThreads
  flags = malloc(N*sizeof(char)); // Flags for checked rows

  //flagtab - global array used as flag table for the checked rows
  flagTab = flags;

  /* Init the matrix	*/
  Init_Matrix();

  /*Creating the threads */
  for (i = 0; i < nproc; i++,id_T[i] = i)
  {
    pthread_create(&pThread[i],NULL,&work,&id_T[i]);
    if(PRINT) printf("Created Thread --> Thread ID : %u\n", id_T[i]);
  }

  /*Terminating the threads*/
  for (i = 0; i < nproc; i++)
  {
    pthread_join(pThread[i],NULL);
    if(PRINT) printf("Finished Thread --> Thread ID : %u\n", id_T[i]);
  }

  if (PRINT)  Print_Matrix();

  /*  Check if all of the threads are terminated and deallocating memory  */
  while(1)
  {
    if(threadCnt == nproc)
    {
      free(flags);
      free(pThread);
      free(id_T);
      flagTab=NULL;
      pThread=NULL;
      return 0;
    }
  }
}

void *
work(void* id_T)
{
  int prevT_Row, cols, currT_Row;
  int currTID = *((unsigned int *)id_T);
  currTID = *((unsigned int *)id_T);

  for (currT_Row=currTID; currT_Row < N;  currT_Row+=nproc) {  /* Current Thread Rows */
    for (prevT_Row = 0; prevT_Row < currT_Row; prevT_Row++) { /*  Previous Thread Rows  */
      pthread_mutex_lock(&lock);
      /* Mutex locked until all of the previous rows
      have already passed the division step and set
      the values in the flag table to 1 */
      while(!flagTab[prevT_Row])
      {
        pthread_cond_wait(&cond,&lock);
      }
      pthread_mutex_unlock(&lock);

      /* Elimination step */
      for (cols = prevT_Row+1; cols < N ; cols++)
      /* Going through the columns after the element on the main diagonal*/
      {
        A[currT_Row][cols] = A[currT_Row][cols] - A[currT_Row][prevT_Row]*A[prevT_Row][cols];
      }
      b[currT_Row] = b[currT_Row] - A[currT_Row][prevT_Row]*y[prevT_Row];

      /*  Setting to 0 the elements under the main diagonal after elimination  */
      A[currT_Row][prevT_Row] =0.0;
    }

    /* DIVISION STEP */
    for (cols = currT_Row+1; cols < N; cols++)
    {
      A[currT_Row][cols] = A[currT_Row][cols] / A[currT_Row][currT_Row];
    }
    y[currT_Row] = b[currT_Row] / A[currT_Row][currT_Row];

    /*  Setting to 1 the elements on the main diagonal after division */
    A[currT_Row][currT_Row] = 1.0;

    pthread_mutex_lock(&lock);
    /* Setting to 1 the value for the current thread row in the flag table
    and sending a signal to the other threads that the current one is finished */
    flagTab[currT_Row]=1;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&lock);
  }
  /* Increasing the counter when the thread finish its work */
  threadCnt++;
}

void
Init_Matrix()
{
  int i, j;

  printf("\nSize of Matrix\t= %dx%d ", N, N);
  printf("\nNumber of Processors\t= %d ", nproc);
  printf("\nMaximum Value\t= %d \n", maxnum);
  printf("Init\t  = %s \n", Init);
  printf("Initializing matrix...\n");

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

  if (PRINT == 1)
  Print_Matrix();
}

void
Print_Matrix()
{
  int i, j;

  printf("\n\nMatrix A:\n");
  for (i = 0; i < N; i++) {
    printf("[");
    for (j = 0; j < N; j++)
    {
      if(j==N-1) printf(" %5.2f", A[i][j]);
      else printf(" %5.2f,", A[i][j]);
    }
    printf("]\n");
  }
  printf("\nVector b:\n[");
  for (j = 0; j < N; j++)
  {
    if(j==N-1) printf(" %5.2f", b[j]);
    else printf(" %5.2f,", b[j]);
  }
  printf("]\n");
  printf("\nVector y:\n[");
  for (j = 0; j < N; j++)
  {
    if(j==N-1) printf(" %5.2f", y[j]);
    else printf(" %5.2f,", y[j]);
  }
  printf("]\n");
  printf("\nVector flagTab : \n[");
  for (j = 0; j < N; j++) {
    {
      if(j==N-1) printf(" %d", flagTab[j]);
      else printf(" %d,", flagTab[j]);
    }
  }
  printf("]\n");
  printf("\ndone \n\n");
}

void
Init_Default()
{
  N = 2048;
  nproc = 8;
  Init = "rand";
  maxnum = 15.0;
  PRINT = 0;
}

int
Read_Options(int argc, char **argv)
{
  char    *prog;

  prog = *argv;
  while (++argv, --argc > 0)
  if (**argv == '-')
  switch ( *++*argv ) {
    case 'N':
    --argc;
    N = atoi(*++argv);
    break;
    case 'n':
    --argc;
    nproc = atoi(*++argv);
    break;
    case 'h':
    printf("\nHELP: try sor -u \n\n");
    exit(0);
    break;
    case 'u':
    printf("\nUsage: sor [-N problemsize]\n");
    printf("           [-n] number of processors \n");
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
