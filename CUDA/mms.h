/*************************************************************

          Sparse Matrix Vector Multiplication 

    ------------------------------------------------------

 Author: LANGE Theo theo.lange.369@cranfield.ac.uk
 Date: 19/02/2023
 
 File: mms.h
 Objective:  This file will define the functions and 
             Storage formats in order to store
             a Sparse Matrix in the CSR and ELLPACK formats
*************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MM_S_H
#define MM_S_H

/********************* Sparse Matrix Storage Formats Definition ***************************/

// Definition of the CSR Storage format
typedef struct {
  int M, N;
  int *IRP, *JA;
  double *AS;
} CSR;

// Definition of the ELLPACK Storage format
// The matrices JA and AS are in row-major order
// i.e JA[i][j] is stored in i * MAXNZ + j element of the vector.
typedef struct {
  int M, N, MAXNZ;
  int *JA;
  double *AS;
} ELLPACK;


/********************* Sparse Matrix Storage Functions Definition ***************************/


/*
Function: Store_CSR()
Objective: This function stores the input sparse matrix into the CSR format

Input: Number of rows and columns (M and N), number of non-zero values, Arrays of rows, columns and values I, J and val
Output: The CSR stored matrix
*/
CSR Store_CSR (int M, int N, int nz, int *I, int *J, double *val);


/*
Function: Store_ELLPACK()
Objective: This function stores the input sparse matrix into the ELLPACK format

Input: Number of rows and columns (M and N), number of non-zero values, Arrays of rows, columns and values I, J and val
Output: The ELLPACK stored matrix
*/
ELLPACK Store_ELLPACK (int M, int N, int nz, int *I, int *J, double *val);


/********************* Sparse Matrix Printing Functions Definition ***************************/

/*
Function: print_CSR()
Objective: Print a Sparse Matrix stored in the CSR format

Input: A Sparse Matrix stored in the CSR format, the number of non-zero values (Integer)
Output: None
*/
void print_CSR (CSR mat, int nz);


/*
Function: print_ELLPACK()
Objective: Print a Sparse Matrix stored in the ELLPACK format

Input: A Sparse Matrix stored in the ELLPACK format, the number of non-zero values (Integer)
Output: None
*/
void print_ELLPACK (ELLPACK mat);

#endif

#ifdef __cplusplus
}
#endif