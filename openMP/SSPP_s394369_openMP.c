/*************************************************************

          Sparse Matrix Vector Multiplication
                OpenMP Unroll 4 Version        

    ------------------------------------------------------

 Author: LANGE Theo s394369 theo.lange.369@cranfield.ac.uk
 Date: 25/02/2023
 
 File: SSPP_s394369_openMP.c
 Objective:  This file will compute the multiplication
             between a Sparse Matrix given as input
             and a random Vector using
             the openMP unroll 4 method
*************************************************************/


/*******************
      Libraries
*******************/

#include <stdlib.h>   // for rand(),srand(), malloc(), realloc() and calloc()
#include <stdio.h>    // for printf()
#include <math.h>     // for fabs()
#include <omp.h>      // for the omp parallel function

#include "wtime.h"   // for the timer
#include "mmio.h"    // for the reading of the input matrices
#include "mms.h"     // for the storage function and storage format structures


/******************************************
  Global variables, functions and types
******************************************/

// Constant integer to compute an average computational time
const int ntimes=1000;

// Definition of the min and max function
inline int max ( int a, int b ) { return a > b ? a : b; }
inline int min ( int a, int b ) { return a < b ? a : b; }

/**************************************

              Functions
      
**************************************/
/*******************
    Serial SpMV
*******************/

/*
Function: CSR_MatrixVector_serial()
Objective: Compute the multiplication between a Sparse Matrix stored in the CSR format and a Vector in serial

Input: A Sparse Matrix stored in the CSR format, a Vector x of size NCOLS and the result vector of size NROWS
Output: None
*/
void CSR_MatrixVector_serial(const CSR mat, const double *x, double* restrict y) 
{
  int i, j;
  double t;
  
  for (i = 0; i < mat.M; ++i) {
    t=0.0;
    for (j = mat.IRP[i]; j < mat.IRP[i+1]; ++j) {
      t +=  mat.AS[j]*x[mat.JA[j]];
    }
    y[i] = t;
    
  }
}

/*
Function: ELLPACK_MatrixVector_serial()
Objective: Compute the multiplication between a Sparse Matrix stored in the ELLPACK format and a Vector in serial

Input: A Sparse Matrix stored in the ELLPACK format, a Vector x of size NCOLS and the result vector of size NROWS
Output: None
*/
void ELLPACK_MatrixVector_serial(const ELLPACK mat, const double *x, double* restrict y) 
{
  int i, j;
  double t;
  
  for (i = 0; i < mat.M; ++i) {
    t=0.0;
    for (j = 0; j < mat.MAXNZ; ++j) {
      t +=  mat.AS[i * mat.MAXNZ + j]*x[mat.JA[i * mat.MAXNZ + j]];
    }
    y[i] = t;
    
  }
}

/*******************
  Parallel SpMV
*******************/


/*
Function: CSR_MatrixVector
Objective:  Compute the multiplication between a Sparse Matrix stored in the CSR format and a Vector in parallel
            using openMP and the unroll 4 method

Input: A Sparse Matrix stored in the CSR format, a Vector x of size NCOLS and the result vector of size NROWS
Output: None
*/
void CSR_MatrixVector(const CSR mat, const double *x, double* restrict y) 
{
  int i, j;

#pragma omp parallel for shared(x,y) private(i,j)
  for (i = 0; i < mat.M - mat.M%4; i += 4) {
    double t0 = 0.0;
    double t1 = 0.0;
    double t2 = 0.0;
    double t3 = 0.0;
  
  
    for (j = mat.IRP[i]; j < mat.IRP[i+1]; j++) t0 += mat.AS[j] * x[mat.JA[j]];
    for (j = mat.IRP[i+1]; j < mat.IRP[i+2]; j++) t1 += mat.AS[j] * x[mat.JA[j]];
    for (j = mat.IRP[i+2]; j < mat.IRP[i+3]; j++) t2 += mat.AS[j] * x[mat.JA[j]];
    for (j = mat.IRP[i+3]; j < mat.IRP[i+4]; j++) t3 += mat.AS[j] * x[mat.JA[j]]; 

      
    y[i+0] = t0;
    y[i+1] = t1;
    y[i+2] = t2;
    y[i+3] = t3;
  }
  
  for (i = mat.M - mat.M%4; i < mat.M; i++) {
    double t = 0.0;
    for (j = mat.IRP[i]; j < mat.IRP[i+1]; j++) {
      t += mat.AS[j]*x[mat.JA[j]];    
    }
    y[i]=t;
  }
}


/*
Function: ELLPACK_MatrixVector
Objective: Compute the multiplication between a Sparse Matrix stored in the ELLPACK format and a Vector in parallel
            using openMP and the unroll 4 method

Input: A Sparse Matrix stored in the ELLPACK format, a Vector x of size NCOLS and the result vector of size NROWS
Output: None
*/
void ELLPACK_MatrixVector(const ELLPACK mat, const double *x, double* restrict y) 
{
  int i, j;
    
  #pragma omp parallel for shared(x,y) private(i,j)
  for (i = 0; i < mat.M - mat.M%4; i += 4) {
    double t0 = 0.0;
    double t1 = 0.0;
    double t2 = 0.0;
    double t3 = 0.0;

    for (j = 0; j < mat.MAXNZ ; j++) {
      t0 += mat.AS[i * mat.MAXNZ + j] * x[mat.JA[i * mat.MAXNZ + j]];
      t1 += mat.AS[(i + 1) * mat.MAXNZ + j] * x[mat.JA[(i + 1) * mat.MAXNZ + j]];
      t2 += mat.AS[(i + 2) * mat.MAXNZ + j] * x[mat.JA[(i + 2) * mat.MAXNZ + j]];
      t3 += mat.AS[(i + 3) * mat.MAXNZ + j] * x[mat.JA[(i + 3) * mat.MAXNZ + j]];

    }
    
    
    y[i+0] = t0;
    y[i+1] = t1;
    y[i+2] = t2;
    y[i+3] = t3;
  }
  
  for (i = mat.M - mat.M%4; i < mat.M; i++) {
    double t=0.0;
    for (j = 0; j < mat.MAXNZ; j++) {
      t = t + mat.AS[i * mat.MAXNZ + j] * x[mat.JA[i * mat.MAXNZ + j]];      
    }
    y[i] = t;
  }
}


/*
Function: main()
Objective:  Read the input file
            Store the matrix in both Storage format
            Compute the Serial SpMV for both Storage Format
            Compute the Parallel SpMV for both Storage Format

Input:  Number of command-line argument (Integer)
        String array containing the command-line argument
Output: Number 0
*/
int main(int argc, char** argv) 
{
  MM_typecode matcode;
  int M, N, nz;   
  int *I, *J;
  double *val;
  CSR CSR_mat;
  ELLPACK ELLPACK_mat;

  // In this case, no input matrix has been given, return an error
  if (argc < 2)
{
	fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
	exit(1);
}   
  
  // If an input file has been given, try to read the matrix.
  if (mm_read_mtx_crd(argv[1], &M, &N, &nz, &I, &J, &val, &matcode)!= 0)
  {
    printf(" Error while Reading the matrix %s\n", argv[1]);
  }
  
  // Print the matrix characterictics
  printf("\nMatrix-vector product: Unroll 4 Version\n");
  printf("Test Matrix: %s\n", argv[1]);
  
  mm_write_banner(stdout, matcode);
  mm_write_mtx_crd_size(stdout, M, N, nz);
  
  printf("\n");
  
  // Store the matrix in both CSR and ELLPACK Format
  CSR_mat = Store_CSR(M, N, nz, I, J, val);
  ELLPACK_mat = Store_ELLPACK(M, N, nz, I, J, val);
  
  

  // Create and allocate memory for all the needed arrays
  double* x = (double*) malloc(sizeof(double)*N);
  
  // Arrays to store the Result of the Serial version of the SpMV
  double* y_true_CSR = (double*) malloc(sizeof(double)*M );
  double* y_true_ELLPACK = (double*) malloc(sizeof(double)*M );
  
  // Arrays to store the Result of the Parallel version of the SpMV
  double* y_CSR = (double*) malloc(sizeof(double)*M );
  double* y_ELLPACK = (double*) malloc(sizeof(double)*M );
  
  // Randomly initialise the Vector x
  srand(12345);
  for (int row = 0; row < M; ++row) {
    x[row] = 100.0f * ((double) rand()) / RAND_MAX;      
  }
  
  double tmlt = 0.0;  // Store the computational time
  double t1, t2;      // Starting and ending time of the SpMV
  double mflops;      // Number of operation per seconds in MFlops
  double bdwdth;      // Memory bandiwidth usage
  
  /**************************************

              Serial SpMV
      
  **************************************/
  
  // Compute the SpMV ntimes
  for (int try=0; try < ntimes; try ++ ) {
    t1 = wtime();
    CSR_MatrixVector_serial(CSR_mat, x, y_true_CSR);
    t2 = wtime();
    tmlt += (t2-t1);
  }
  
  // Determine the average computational time and the performance in MFLOPS
  tmlt /= ntimes;
  mflops = (2.0e-6)*nz/tmlt;
  
  fprintf(stdout,"CSR: Serial Matrix-Vector product of size %d x %d: time %lf  MFLOPS %lf \n",
	      M,N,tmlt,mflops);
         
  
  tmlt = 0.0;

  // Compute the SpMV ntimes
  for (int try=0; try < ntimes; try ++ ) {
    t1 = wtime();
    ELLPACK_MatrixVector_serial(ELLPACK_mat, x, y_true_ELLPACK);
    t2 = wtime();
    tmlt += (t2-t1);
  }
  
  // Determine the average computational time and the performance in MFLOPS
  tmlt /= ntimes;
  mflops = (2.0e-6)*nz/tmlt;
  
  fprintf(stdout,"ELLPACK: Serial Matrix-Vector product of size %d x %d: time %lf s MFLOPS %lf\n\n",
	      M,N,tmlt,mflops);
  
  /**************************************

              Parallel SpMV
      
  **************************************/
  
  tmlt = 0.0;
  
  // Compute the SpMV ntimes
  for (int try=0; try < ntimes; try ++ ) {
    t1 = wtime();
    CSR_MatrixVector(CSR_mat, x, y_CSR);
    t2 = wtime();
    tmlt += (t2-t1);
  }
  
  // Determine the average computational time and the performance in MFLOPS
  tmlt /= ntimes;
  mflops = (2.0e-6)*nz/tmlt;
  bdwdth = (12.0e-9)*((double) nz * sizeof(double))/tmlt;
  
#pragma omp parallel 
  {
#pragma omp master
    {
      fprintf(stdout,"CSR Matrix-Vector product (unroll_4) of size %d x %d with %d threads: time %lf s MFLOPS %lf Memory Bandwidth Usage %lf GB/s \n",
	      M,N,omp_get_num_threads(),tmlt,mflops,bdwdth);
    }
  }
  
  // Compare the result between the Serial and Parallel SpMV using the CSR Storage Format
  double reldiff = 0.0f;
  double diff = 0.0f;
  
  for (int row = 0; row < M; ++row) {
    double maxabs = max(fabs(y_CSR[row]),fabs(y_true_CSR[row]));
    if (maxabs == 0.0) maxabs=1.0;
    reldiff = max(reldiff, fabs(y_CSR[row] - y_true_CSR[row])/maxabs);
    diff = max(diff, fabs(y_CSR[row] - y_true_CSR[row]));
  }
  
  printf("Max diff = %lf, Max rel diff = %lf\n\n", diff, reldiff);
  
  tmlt = 0.0;

  // Compute the SpMV ntimes
  for (int try=0; try < ntimes; try ++ ) {
    t1 = wtime();
    ELLPACK_MatrixVector(ELLPACK_mat, x, y_ELLPACK);
    t2 = wtime();
    tmlt += (t2-t1);
  }
  
  // Determine the average computational time and the performance in MFLOPS
  tmlt /= ntimes;
  mflops = (2.0e-6)*nz/tmlt;
  bdwdth = (12.0e-9)*((double) nz * sizeof(double))/tmlt;
  
#pragma omp parallel 
  {
#pragma omp master
    {
      fprintf(stdout,"ELLPACK Matrix-Vector product (unroll_4) of size %d x %d with %d threads: time %lf s MFLOPS %lf Memory Bandwidth Usage %lf GB/s \n",
	      M,N,omp_get_num_threads(),tmlt,mflops,bdwdth);
    }
  }
  
  // Compare the result between the Serial and Parallel SpMV using the CSR Storage Format
  reldiff = 0.0f;
  diff = 0.0f;
  
  for (int row = 0; row < M; ++row) {
    double maxabs = max(fabs(y_ELLPACK[row]),fabs(y_true_ELLPACK[row]));
    if (maxabs == 0.0) maxabs=1.0;
    reldiff = max(reldiff, fabs(y_ELLPACK[row] - y_true_ELLPACK[row])/maxabs);
    diff = max(diff, fabs(y_ELLPACK[row] - y_true_ELLPACK[row]));
  }
  
  printf("Max diff = %lf, Max rel diff = %lf\n", diff, reldiff);
  
  
  // Free the memory
  free(x);
  free(y_true_CSR);
  free(y_true_ELLPACK);
  free(y_CSR);
  free(y_ELLPACK);
  return 0;
}
