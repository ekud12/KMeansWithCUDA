
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"

Point* points_array_on_coda_device;
int cuda_number_of_points , block_number , threads_per_block;

__global__ void calculatPointsLocation(Point* points_array_on_cuda, float dt , float maxT , float pi , int array_size , int blockSize);

//initialize cuda device with error checking and props
void initCodaDevice()
{
	cudaError_t cudaStatus;
	cudaDeviceProp deviceProp;

	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return;
    }

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return;
	}
	
    cudaGetDeviceProperties(&deviceProp, 0);
	threads_per_block = deviceProp.maxThreadsPerBlock;
}


void copyPointArrayToCudaDevice(Point* pointsArray , int size)
{
	cudaError_t cudaStatus;
	cuda_number_of_points = size;


	cudaStatus = cudaMalloc((void**)&points_array_on_coda_device, cuda_number_of_points * sizeof(Point));
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaMalloc for cuda_points_array failed!");
		releaseCudaMemory();
        return;

    }

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		releaseCudaMemory();
        return;
	}

	cudaStatus = cudaMemcpy(points_array_on_coda_device, pointsArray, cuda_number_of_points * sizeof(Point), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy failed!");
        releaseCudaMemory();
        return;
    }

	block_number = 1;
	while(block_number * threads_per_block < cuda_number_of_points)
	{
		block_number++;
	}

	while(block_number * threads_per_block > cuda_number_of_points)
	{
		threads_per_block--;
	}

	if(block_number * threads_per_block < cuda_number_of_points)
		threads_per_block++;


	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		releaseCudaMemory();
        return;
	}
}

//new points calculation Wrapper function
void movePointsLocationWithCuda(Point* pointsArray , float dt, float maxT , float pi)
{
	cudaError_t cudaStatus;

	//send all relevent details
	calculatPointsLocation<<<block_number, threads_per_block>>>(points_array_on_coda_device , dt ,  maxT , pi ,cuda_number_of_points ,threads_per_block);

	//error checking
	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "calculatPointsCoordinatePerThread launch failed: %s\n", cudaGetErrorString(cudaStatus));
        releaseCudaMemory();
        return;
    }

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		releaseCudaMemory();
        return;
	}

	//copy array back to host
	cudaStatus = cudaMemcpy(pointsArray, points_array_on_coda_device, cuda_number_of_points * sizeof(Point), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy failed!");
        releaseCudaMemory();
        return;
    }

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		releaseCudaMemory();
        return;
	}
}

//release the Cuda Memory
void releaseCudaMemory()
{
	cudaFree(points_array_on_coda_device);
}

//calculate new points location on cuda array
__global__ void calculatPointsLocation(Point* points_array_on_cuda, float dt , float maxT , float pi , int array_size , int blockSize)
{
	float calculation = (2 * pi * dt) / maxT;
	int thread_id = threadIdx.x;
	int block_id = blockIdx.x;
	int position;

	position = block_id * blockSize + thread_id;

	if(position < array_size)
	{
		points_array_on_cuda[position].x = points_array_on_cuda[position].a + points_array_on_cuda[position].r * cos(calculation);
		points_array_on_cuda[position].y = points_array_on_cuda[position].b + points_array_on_cuda[position].r * sin(calculation);
	}
}