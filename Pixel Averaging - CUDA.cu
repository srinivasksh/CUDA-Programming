//Submission should be named as  <RollNumber>_Prog.cu
//Upload just this cu file and nothing else. If you upload it as a zip, it will not be evaluated. 


#include <stdio.h>
#define M 514 
//Input has 514 rows and columns 

#define N 512 
//For output, only 512 rows and columns need to be computed. 


//TODO: WRITE GPU KERNEL. It should not be called repeatedly from the host, but just once. Each time it is called, it may process more than pixel or not //process any pixel at all. 

//Code to proces 514*514 input elements using 9*48 threads
//Additional checks included to skip processing boundary elements (only 512*512 to be computed)
// i> 513 : Skip Top row ; i < 263682 : Skip Bottom row 
// i%514=0 : Skip left most column ; (i+1)%514=0 : Skip Right most column

__global__ void computeOutput( int *a, int *b, int size) {
	int numThrds = 9*48;
	int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i=threadID;i<size;i+=numThrds) {
		if ((i > 513) && (i < 263682 ) && (i%514 != 0 ) & ((i+1)%514 != 0) ){
			b[i] = (a[i-1]+a[i+1]+a[i-514]+a[i+514])/4;
		}
	}
}

main (int argc, char **argv) {
  int A[M][M], B[M][M];
  int *d_A, *d_B; // These are the copies of A and B on the GPU
  int *h_B;       // This is a host copy of the output of B from the GPU
  int i, j;

  // Input is randomly generated
  for(i=0;i<M;i++) {
    for(j=0;j<M;j++) {
      A[i][j] = rand()/1795831;
      //printf("%d\n",A[i][j]);
    }
  }

  // sequential implementation of main computation
  for(i=1;i<M-1;i++) {
    for(j=1;j<M-1;j++) {
      B[i][j] = (A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4;
    }
  }


  // TODO: ALLOCATE MEMORY FOR GPU COPIES OF d_A AND d_B
  cudaMalloc((void **) &d_A, M*M*sizeof(int));
  cudaMalloc((void **) &d_B, M*M*sizeof(int));
  
  int insize = M*M*sizeof(int);

  // TODO: COPY A TO d_A
  cudaMemcpy(d_A, A, insize, cudaMemcpyHostToDevice);

  // TODO: CREATE BLOCKS with THREADS AND INVOKE GPU KERNEL
  //Use 9 blocks, each with 48 threads
  computeOutput<<<9,48>>>(d_A,d_B,M*M);
  cudaDeviceSynchronize();

  // TODO: COPY d_B BACK FROM GPU to CPU in variable h_B
  h_B = (int *) malloc(insize);
  cudaMemcpy(h_B, d_B, insize, cudaMemcpyDeviceToHost);

  // TODO: Verify result is correct by comparing
  int error_cnt=0;
  for(i=1;i<M-1;i++) {
    for(j=1;j<M-1;j++) {
		//print only those elements for which the above subtraction is non-zero
		if (B[i][j] - h_B[i*514+j] != 0) {
			printf("i: %d ,j: %d , B[i][j]: %d , h_B[i*514+j]: %d \n", i,j,B[i][j],h_B[i*514+j]);
			error_cnt=error_cnt+1;
		}
    }
  }
  

  //IF even one element of h_B and B differ, report an error.
  //Otherwise, there is no error.
  //If your program is correct, no error should occur.
  if (error_cnt>1) {
	printf("Error: Found %d discrepancies between CPU and GPU calculation \n", error_cnt);
  } 
  else {
	printf("Code completed successfully! \n");
  }	
}
   

/*Remember the following guidelines to avoid losing marks
Index of an array should not exceed the array size. 
Do not ignore the fact that boundary rows and columns need not be computed (in fact, they cannot be computed since they don't have four neighbors)
No output array-element should be computed more than once
No marks will be given if the program does not compile or run (TAs will not debug your program at all)
*/

