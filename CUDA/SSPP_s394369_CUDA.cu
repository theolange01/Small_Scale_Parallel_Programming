/*************************************************************

          Sparse Matrix Vector Multiplication
                CUDA 2D block version        

    ------------------------------------------------------

 Author: LANGE Theo s394369 theo.lange.369@cranfield.ac.uk
 Date: 25/02/2023
 
 File: SSPP_s394369_CUDA.cu
 Objective:  This file will compute the multiplication
             between a Sparse Matrix given as input
             and a random Vector on GPUs using CUDA on 2D
             blocks of threads
*************************************************************/

/*******************
      Libraries
*******************/

#include <iostream>  // for cin and cout

#include "mmio.h"    // for the reading of the input matrices
#include "mms.h"     // for the storage function and storage format structures

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>   // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers


/******************************************
              Block Dimension
******************************************/

#define XBD 16
#define YBD 8
const dim3 BLOCK_DIM(XBD,YBD);


/**************************************

              Functions
      
**************************************/

/*******************
    Serial SpMV
*******************/

/*
Function: CSR_CPUMatrixVector()
Objective: Compute the multiplication between a Sparse Matrix stored in the CSR format and a Vector in serial on CPUs

Input: A Sparse Matrix stored in the CSR format, a Vector x of size NCOLS and the result vector of size NROWS
Output: None
*/
void CSR_CPUMatrixVector(const CSR mat, const double *x, double* y) 
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
Function: ELLPACK_CPUMatrixVector()
Objective: Compute the multiplication between a Sparse Matrix stored in the ELLPACK format and a Vector in serial on CPUs

Input: A Sparse Matrix stored in the ELLPACK format, a Vector x of size NCOLS and the result vector of size NROWS
Output: None
*/
void ELLPACK_CPUMatrixVector(const ELLPACK mat, const double *x, double* y) 
{
  int i, j;
  double t;
  
  for (i = 0; i < mat.M; ++i) {
    t=0.0;
    for (j = 0; j < mat.MAXNZ; ++j) {
      t = t + mat.AS[i * mat.MAXNZ + j]*x[mat.JA[i * mat.MAXNZ + j]];
    }
    y[i] = t;
  }
}

/*******************
  Parallel SpMV
*******************/

/*
Function: rowReduce()
Objective: 

Input: A shared double array, index tid and integer s
Output: None
*/
__device__ void rowReduce(volatile double *sdata, int tid, int s) {
  switch(s){
  case 16:  sdata[tid] += sdata[tid + 16];
  case  8:  sdata[tid] += sdata[tid +  8];
  case  4:  sdata[tid] += sdata[tid +  4];
  case  2:  sdata[tid] += sdata[tid +  2];
  case  1:  sdata[tid] += sdata[tid +  1];
  }
}


/*
Function: CSR_gpuMatrixVector()
Objective: Compute the matrix_vector product in the CSR format on GPUs using 2D blocks of threads

Input: Number of rows (nrows), of columns (ncols), arrays IRP, JA and AS of the CSR format, dense Vector x, result vector y
Output: None
*/
__global__ void CSR_gpuMatrixVector(int nrows, int ncols, const int* IRP, const int* JA, const double* AS, const double* x, double* y) {
  __shared__ double aux[YBD][XBD];
  int tr     = threadIdx.y;
  int tc     = threadIdx.x;
  int row    = blockIdx.x*blockDim.y + tr;
  int s;
  aux[tr][tc] = 0.0;
  
  if (row < nrows) {
    double t = 0.0;
    for (int j = IRP[row] + tc; j < IRP[row+1]; j+= XBD) {
      t +=  AS[j]*x[JA[j]];
    }
    aux[tr][tc] = t;
  }
  __syncthreads();
  
  for (int s=XBD/2; s >= 32; s >>=1) {
    if (tc<s) aux[tr][tc] += aux[tr][tc+s];
    __syncthreads();
  }
  
  s = min(16, XBD/2);
  if (tc < s) rowReduce(&(aux[tr][0]), tc, s);
  
  if ((tc == 0) && (row<nrows)) y[row] = aux[tr][tc];
}


/*
Function: ELLPACK_gpuMatrixVector()
Objective: Compute the matrix_vector product in the ELLPACK format on GPUs using 2D blocks of threads

Input: Number of rows (nrows), of columns (ncols), arrays IRP, JA and AS of the CSR format, dense Vector x, result vector y
Output: None
*/
__global__ void ELLPACK_gpuMatrixVector(int rows, int cols, int maxnz, const int* JA, const double* AS, const double* x, double* y) {
  __shared__ double aux[YBD][XBD];
  
  int tr     = threadIdx.y;
  int tc     = threadIdx.x;
  int row    = blockIdx.x*blockDim.y + tr;
  int s;
  aux[tr][tc] = 0.0;
  
  if (row < rows) {
    double t = 0.0;

    for (int j = tc; j < maxnz; j += XBD) {
      t +=  AS[row * maxnz + j]*x[JA[row * maxnz + j]];
    }
    
    aux[tr][tc] = t;
  }
  __syncthreads();
  
  for (int s=XBD/2; s >= 32; s >>=1) {
    if (tc<s) aux[tr][tc] += aux[tr][tc+s];
    __syncthreads();
  }
  
  s = min(16, XBD/2);
  if (tc < s) rowReduce(&(aux[tr][0]), tc, s);
  
  if ((tc == 0) && (row<rows)) y[row] = aux[tr][tc];
}


/*
Function: main()
Objective:  Read the input file
            Store the matrix in both Storage format
            Compute the Serial SpMV for both Storage Format
            Compute the Parallel SpMV for both Storage Format on GPU

Input:  Number of command-line argument (Integer)
        String array containing the command-line argument
Output: Number 0
*/
int main(int argc, char** argv) {

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
  std::cout << "\nMatrix-vector product: 2D thread block version "  <<std::endl;
  std::cout << "Test case: " << argv[1] << std::endl;
  std::cout << std::endl;
  
  mm_write_banner(stdout, matcode);
  mm_write_mtx_crd_size(stdout, M, N, nz);
  
  std::cout << std::endl;
  
  
  // Store the matrix in both CSR and ELLPACK Format
  CSR_mat = Store_CSR(M, N, nz, I, J, val);
  ELLPACK_mat = Store_ELLPACK(M, N, nz, I, J, val);
  
  // ----------------------- Host memory initialisation ----------------------- //

  double* h_x = new double[N];
  double* CSR_h_y = new double[M];
  double* ELLPACK_h_y = new double[M];
  double* CSR_h_y_d = new double[M];
  double* ELLPACK_h_y_d = new double[M];

  srand(123456);
  for (int row = 0; row < M; ++row) {
    CSR_h_y[row] = 0.0;
    ELLPACK_h_y[row] = 0.0;
  }
  for (int col = 0; col < N; ++col) {
    h_x[col] = 100.0f * static_cast<double>(rand()) / RAND_MAX;
  }
// ---------------------- Device memory initialisation ---------------------- //
  //  Allocate memory space on the device. 
  
  double *d_x, *CSR_d_y, *ELLPACK_d_y;

  checkCudaErrors(cudaMalloc((void**) &d_x, N * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**) &CSR_d_y, M * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**) &ELLPACK_d_y, M * sizeof(double)));

  // Copy matrices from the host (CPU) to the device (GPU).
  checkCudaErrors(cudaMemcpy(d_x, h_x,  N * sizeof(double), cudaMemcpyHostToDevice));
  
  /********CSR**********/
  
  int *CSR_d_IRP, *CSR_d_JA;
  double *CSR_d_AS;
  
  checkCudaErrors(cudaMalloc((void**) &CSR_d_IRP, (M + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &CSR_d_JA, nz * sizeof(int)));
  
  checkCudaErrors(cudaMalloc((void**) &CSR_d_AS, nz * sizeof(double)));
  
  // Copy matrices from the host (CPU) to the device (GPU).
  checkCudaErrors(cudaMemcpy(CSR_d_IRP, CSR_mat.IRP,  (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(CSR_d_JA, CSR_mat.JA,  nz * sizeof(int), cudaMemcpyHostToDevice));
  
  checkCudaErrors(cudaMemcpy(CSR_d_AS, CSR_mat.AS,  nz * sizeof(double), cudaMemcpyHostToDevice));
  
  /********ELLPACK**********/
  
  int *ELLPACK_d_JA;
  double *ELLPACK_d_AS;

  checkCudaErrors(cudaMalloc((void **) &ELLPACK_d_JA, M * ELLPACK_mat.MAXNZ * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **) &ELLPACK_d_AS, M * ELLPACK_mat.MAXNZ * sizeof(double)));
  
  // Copy matrices from the host (CPU) to the device (GPU).
  checkCudaErrors(cudaMemcpy(ELLPACK_d_JA, ELLPACK_mat.JA,  M * ELLPACK_mat.MAXNZ * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ELLPACK_d_AS, ELLPACK_mat.AS,  M * ELLPACK_mat.MAXNZ * sizeof(double), cudaMemcpyHostToDevice));                 
  
  

  // ------------------------ Calculations on the CPU ------------------------- //
  float flopcnt=2.e-6*nz;
  
  
  // Create the CUDA SDK timer.
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);

  timer->start();
  CSR_CPUMatrixVector(CSR_mat, h_x, CSR_h_y);

  timer->stop();
  float cpuflops=flopcnt/ timer->getTime();
  std::cout << "CSR  CPU time: " << timer->getTime() << " ms." << " GFLOPS " << cpuflops << std::endl;
  
  timer->reset();
  timer->start();
  ELLPACK_CPUMatrixVector(ELLPACK_mat, h_x, ELLPACK_h_y);

  timer->stop();
  cpuflops=flopcnt/ timer->getTime();
  std::cout << "ELLPACK  CPU time: " << timer->getTime() << " ms." << " GFLOPS " << cpuflops << std::endl;
// ------------------------ Calculations on the GPU ------------------------- //

  //--------------------------CSR---------------------------//
  
  
  // Calculate the dimension of the grid of blocks (1D) necessary to cover
  // all rows. 
  const dim3 GRID_DIM((M - 1 + BLOCK_DIM.y)/ BLOCK_DIM.y,1);	
  double bdwdth;
  
  timer->reset();
  timer->start();
  
  CSR_gpuMatrixVector<<<GRID_DIM, BLOCK_DIM >>>(M, N, CSR_d_IRP, CSR_d_JA, CSR_d_AS, d_x, CSR_d_y);
  checkCudaErrors(cudaDeviceSynchronize());

  timer->stop();
  float gpuflops=flopcnt/ timer->getTime();
  bdwdth = (12.0e-6)*((double) nz * sizeof(double))/timer->getTime();
  std::cout << "CSR  GPU time: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<< " Memory Bandwidth Usage " << bdwdth<< " GB/s" << std::endl;

  // Download the resulting vector d_y from the device and store it in h_y_d.
  checkCudaErrors(cudaMemcpy(CSR_h_y_d, CSR_d_y, M*sizeof(double),cudaMemcpyDeviceToHost));


  // Now let's check if the results are the same.
  double reldiff = 0.0f;
  double diff = 0.0f;
  
  for (int row = 0; row < M; ++row) {
    double maxabs = std::max(std::abs(CSR_h_y[row]),std::abs(CSR_h_y_d[row]));
    if (maxabs == 0.0) maxabs=1.0;
    reldiff = std::max(reldiff, std::abs(CSR_h_y[row] - CSR_h_y_d[row])/maxabs);
    diff = std::max(diff, std::abs(CSR_h_y[row] - CSR_h_y_d[row]));
  }
  std::cout << "Max diff = " << diff << "  Max rel diff = " << reldiff << std::endl;
  std::cout << std::endl;
  
  //--------------------------ELLPACK--------------------------//
    
  timer->reset();
  timer->start();
  ELLPACK_gpuMatrixVector<<<GRID_DIM, BLOCK_DIM >>>(M, N, ELLPACK_mat.MAXNZ, ELLPACK_d_JA, ELLPACK_d_AS, d_x, ELLPACK_d_y);
  checkCudaErrors(cudaDeviceSynchronize());

  timer->stop();
  gpuflops=flopcnt/ timer->getTime();
  bdwdth = (12.0e-6)*((double) nz * sizeof(double))/timer->getTime();
  std::cout << "ELLPACK GPU time: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<< " Memory Bandwidth Usage " << bdwdth<< " GB/s" << std::endl;

  // Download the resulting vector d_y from the device and store it in h_y_d.
  checkCudaErrors(cudaMemcpy(ELLPACK_h_y_d, ELLPACK_d_y, M*sizeof(double),cudaMemcpyDeviceToHost));

  // Now let's check if the results are the same.
  
  reldiff = 0.0f;
  diff = 0.0f;
  
  for (int row = 0; row < M; ++row) {
    double maxabs = std::max(std::abs(ELLPACK_h_y[row]),std::abs(ELLPACK_h_y_d[row]));
    if (maxabs == 0.0) maxabs=1.0;
    reldiff = std::max(reldiff, std::abs(ELLPACK_h_y[row] - ELLPACK_h_y_d[row])/maxabs);
    diff = std::max(diff, std::abs(ELLPACK_h_y[row] - ELLPACK_h_y_d[row]));
  }
  std::cout << "Max diff = " << diff << "  Max rel diff = " << reldiff << std::endl;

// ------------------------------- Cleaning up ------------------------------ //

  delete timer;

  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(CSR_d_y));
  checkCudaErrors(cudaFree(ELLPACK_d_y));
  
  checkCudaErrors(cudaFree(ELLPACK_d_JA));
  checkCudaErrors(cudaFree(ELLPACK_d_AS));

  delete[] h_x;
  delete[] CSR_h_y;
  delete[] ELLPACK_h_y;
  delete[] CSR_h_y_d;
  delete[] ELLPACK_h_y_d;
  
  delete[] CSR_mat.JA;
  delete[] CSR_mat.IRP;
  delete[] CSR_mat.AS;
  
  delete[] ELLPACK_mat.JA;
  delete[] ELLPACK_mat.AS;

  delete[] I;
  delete[] J;
  delete[] val;
  return 0;
}
