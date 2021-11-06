#include <stdio.h>

#define TILE_SIZE 32

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    __shared__ float M[TILE_SIZE][TILE_SIZE];
    __shared__ float N[TILE_SIZE][TILE_SIZE];
	
    int tx,ty,row,col;
    float Cv=0.0;
    
    tx=threadIdx.x;
    ty=threadIdx.y;
    row=blockIdx.y*blockDim.y+threadIdx.y;
    col=blockIdx.x*blockDim.x+threadIdx.x;
	
    for(int i=0; i<(k-1)/TILE_SIZE+1; ++i)
    {
      if(i*TILE_SIZE+tx<k && row<m)
	    M[ty][tx]=A[row*k+i*TILE_SIZE+tx];
      else
	    M[ty][tx]=0.0; 
	    
      if(i*TILE_SIZE+ty<k&&col<n)
	    N[ty][tx]=B[col+(i*TILE_SIZE+ty)*n];
      else
	    N[ty][tx]=0.0;
	    
     __syncthreads();
	    
    for(int j=0; j<TILE_SIZE; ++j)
	    Cv+=M[ty][j]*N[j][tx];
	    
    __syncthreads();
  
    }
	
    if(row<m&&col<n)
	    C[row*n+col]=Cv;
	
    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 dimGrid((n-1)/BLOCK_SIZE+1,(m-1)/BLOCK_SIZE+1,1);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
    mysgemm <<< dimGrid, dimBlock >>>(m, n, k, A, B, C);
    /*************************************************************************/
}


