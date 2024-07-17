/*************************************************************

          Sparse Matrix Vector Multiplication

    ------------------------------------------------------

 Author: LANGE Theo theo.lange.369@cranfield.ac.uk
 Date: 19/02/2023
 
 File: mms.c
 Objective:  This file will define the functions and 
             Storage formats in order to store
             a Sparse Matrix in the CSR and ELLPACK formats
*************************************************************/

/*******************
      Libraries
*******************/

#include <stdio.h>   // for printf()
#include <stdlib.h>  // for malloc(), calloc(), realloc() and free()

#include "mms.h"


/********************* Sparse Matrix Storage functions ***************************/

/*
Function: Store_CSR()
Objective: This function stores the input sparse matrix into the CSR format

Input: Number of rows and columns (M and N), number of non-zero values, Arrays of rows, columns and values I, J and val
Output: The CSR stored matrix
*/
CSR Store_CSR (int M, int N, int nz, int *I, int *J, double *val) {
  CSR CSR_mat;
  int i, j, *padding;
  
  CSR_mat.M = M; CSR_mat.N = N; // Store the size of the matrix
  
  // Allocate memory space
  CSR_mat.IRP = (int *) calloc((M+1), sizeof(int));
  padding = (int *) calloc(M, sizeof(int));
  CSR_mat.JA = (int *) malloc(nz * sizeof(int));
  CSR_mat.AS = (double *) malloc(nz * sizeof(double));
  
  // Add 1 to the element row + 1 each time there is an element in the row row
  for (i = 0; i<nz; i ++) CSR_mat.IRP[I[i] + 1] += 1;
  
  // Compute the cumulative sum to obtain IRP
  for (i = 1; i<M+1; i ++) CSR_mat.IRP[i] += CSR_mat.IRP[i-1];
  
  // Given IRP and the number of value already read in the row (padding), fill JA and AS
  for (i = 0; i < nz; i++) {
    int src = CSR_mat.IRP[I[i]] + padding[I[i]];
    CSR_mat.JA[src] = J[i];
    CSR_mat.AS[src] = val[i];
    padding[I[i]] += 1;
  }
  
  // Remove the memory space for this array
  free(padding);
  
  // Return the input Sparse Matrix in the CSR format
  return CSR_mat;
}

/*
Function: Store_ELLPACK()
Objective: This function stores the input sparse matrix into the ELLPACK format

Input: Number of rows and columns (M and N), number of non-zero values, Arrays of rows, columns and values I, J and val
Output: The ELLPACK stored matrix
*/
ELLPACK Store_ELLPACK (int M, int N, int nz, int *I, int *J, double *val) {
  ELLPACK ELLPACK_mat;
  int *nb_nz, *padding;
  int i, j;
  
  ELLPACK_mat.M = M; ELLPACK_mat.N = N; // Store the size of the matrix
  
  // Compute the number of nnz per row to determine MAXNZ
  nb_nz = (int *) calloc(M, sizeof(int));
  for (i = 0;i<nz;i++) nb_nz[I[i]] += 1;
  
  ELLPACK_mat.MAXNZ = nb_nz[0];
  for (i = 1; i<M; i++) {
    if (ELLPACK_mat.MAXNZ < nb_nz[i]) ELLPACK_mat.MAXNZ = nb_nz[i];
  }
  
  // Memory Allocation
  padding = (int *) calloc(M, sizeof(int));
  ELLPACK_mat.JA = (int *) calloc(M * ELLPACK_mat.MAXNZ, sizeof(int));  
  ELLPACK_mat.AS = (double *) calloc(M * ELLPACK_mat.MAXNZ, sizeof(double));
  
  // Given the number of value already stored in a specific row (padding), fill JA and AS
  for (i = 0; i < nz; i++) {
    ELLPACK_mat.JA[I[i] * ELLPACK_mat.MAXNZ + padding[I[i]]] = J[i];
    ELLPACK_mat.AS[I[i] * ELLPACK_mat.MAXNZ + padding[I[i]]] = val[i];
    padding[I[i]] += 1; // Update the number of value stored in the row I[i]
    
    if (padding[I[i]] == nb_nz[I[i]] < ELLPACK_mat.MAXNZ) { // In this case, the row has less nnz than MAXNZ
      // The next elements will have the same value as the last known one
      for (j = nb_nz[I[i]]; j < ELLPACK_mat.MAXNZ; j++) {
        ELLPACK_mat.JA[I[i] * ELLPACK_mat.MAXNZ + j] = ELLPACK_mat.JA[I[i] * ELLPACK_mat.MAXNZ + padding[I[i]] - 1];
      }
    }
  }
  
  // Free the useless arrays
  free(nb_nz);
  free(padding);
  
  // Return the input Sparse Matrix in the ELLPACK format
  return ELLPACK_mat;
}


/********************* Sparse Matrix Printing functions ***************************/

/*
Function: print_CSR()
Objective: Print a Sparse Matrix stored in the CSR format

Input: A Sparse Matrix stored in the CSR format, the number of non-zero values (Integer)
Output: None
*/
void print_CSR (CSR mat, int nz) {
  printf("M = %d\nN = %d\nIRP = [", mat.M, mat.N);
  
  for (int i=0;i<mat.M;i++) {
    printf("%d, ", mat.IRP[i]);
  }
  printf("%d]\nJA = [", mat.IRP[mat.M]);
  
  for (int i=0;i<nz-1;i++) {
    printf("%d, ", mat.JA[i]);
  }
  printf("%d]\nAS = [", mat.JA[nz-1]);
  
  for (int i=0;i<nz-1;i++) {
    printf("%lf, ", mat.AS[i]);
  }
  printf("%lf]\n", mat.AS[nz-1]);
}


/*
Function: print_ELLPACK()
Objective: Print a Sparse Matrix stored in the ELLPACK format

Input: A Sparse Matrix stored in the ELLPACK format, the number of non-zero values (Integer)
Output: None
*/
void print_ELLPACK (ELLPACK mat) {
  printf("M = %d\nN = %d\nMAXNZ = %d\nJA = \n", mat.M, mat.N, mat.MAXNZ);
  
  for (int i=0;i<mat.M;i++) {
    for (int j=0;j<mat.MAXNZ;j++) {
      printf("%d ", mat.JA[i * mat.MAXNZ + j]);
    }
    printf("\n");
  }
  
  printf("\nAS = \n");
  for (int i=0;i<mat.M;i++) {
    for (int j=0;j<mat.MAXNZ;j++) {
      printf("%lf ", mat.AS[i * mat.MAXNZ + j]);
    }
    printf("\n");
  }
}