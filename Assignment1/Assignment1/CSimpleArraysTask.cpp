/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CSimpleArraysTask.h"

#include "../Common/CLUtil.h"

#include <string.h>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CSimpleArraysTask

CSimpleArraysTask::CSimpleArraysTask(size_t ArraySize)
	: m_ArraySize(ArraySize)
{
}

CSimpleArraysTask::~CSimpleArraysTask()
{
	ReleaseResources();
}

bool CSimpleArraysTask::InitResources(cl_device_id Device, cl_context Context)
{
	//CPU resources
	m_hA = new int[m_ArraySize];
	m_hB = new int[m_ArraySize];
	m_hC = new int[m_ArraySize];
	m_hGPUResult = new int[m_ArraySize];
	
	//fill A and B with random integers
	for(unsigned int i = 0; i < m_ArraySize; i++)
	{
		m_hA[i] = rand() % 1024;
		m_hB[i] = rand() % 1024;
	}

	//device resources

	/////////////////////////////////////////
	// Sect. 4.5

	cl_int clError;
	// buffers are declared in header
	m_dA = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(cl_int) * m_ArraySize, NULL, &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create buffer on device");
	m_dB = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(cl_int) * m_ArraySize, NULL, &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create buffer on device");
	m_dC = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * m_ArraySize, NULL, &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create buffer on device");


	/////////////////////////////////////////
	// Sect. 4.6.
	
	// load and compile kernels

	// size_t programSize = 0; 	// unused
	string programCode;

	// create program object (this might contain multiple kernel entry points)
	if (!CLUtil::LoadProgramSourceToMemory("../Assignment1/VectorAdd.cl", programCode)) {
			return false;
	}
	m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
	if (m_Program == nullptr) return false;

	//create kernels from program
	m_Kernel = clCreateKernel(m_Program, "VecAdd", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: VecAdd");


	//bind kernel arguments: adresses of vectors and size of them
	clError  = clSetKernelArg(m_Kernel, 0, sizeof(cl_mem), (void*) &m_dA); 
	clError |= clSetKernelArg(m_Kernel, 1, sizeof(cl_mem), (void*) &m_dB);
	clError |= clSetKernelArg(m_Kernel, 2, sizeof(cl_mem), (void*) &m_dC);
	clError |= clSetKernelArg(m_Kernel, 3, sizeof(cl_int), (void*) &m_ArraySize);
	V_RETURN_FALSE_CL(clError, "Failed to set kernel args: VecAdd");


	return true;
}

void CSimpleArraysTask::ReleaseResources()
{
	//CPU resources
	SAFE_DELETE_ARRAY(m_hA);
	SAFE_DELETE_ARRAY(m_hB);
	SAFE_DELETE_ARRAY(m_hC);
	SAFE_DELETE_ARRAY(m_hGPUResult);

	/////////////////////////////////////////////////
	// Sect. 4.5., 4.6.	

	// free resources on the GPU
	cl_int clErr = 0;
	//clReleaseMemObject(m_dA);
	SAFE_RELEASE_MEMOBJECT(m_dA);
	V_RETURN_CL(clErr, "Could not release memory A on device.");
	//clReleaseMemObject(m_dB);
	SAFE_RELEASE_MEMOBJECT(m_dB);
	V_RETURN_CL(clErr, "Could not release memory B on device.");
	//clReleaseMemObject(m_dC);
	SAFE_RELEASE_MEMOBJECT(m_dC);
	V_RETURN_CL(clErr, "Could not release memory C on device.");

	// release program
	SAFE_RELEASE_PROGRAM(m_Program);
	// release kernel
	SAFE_RELEASE_KERNEL(m_Kernel);


}

void CSimpleArraysTask::ComputeCPU()
{
	for(unsigned int i = 0; i < m_ArraySize; i++)
	{
		m_hC[i] = m_hA[i] + m_hB[m_ArraySize - i - 1];
	}
}

void CSimpleArraysTask::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	/////////////////////////////////////////////////
	// Sect. 4.5
	// Write input data to the GPU
	cl_int clErr;
	clErr = clEnqueueWriteBuffer(CommandQueue, m_dA, CL_FALSE, 0, m_ArraySize * sizeof(int), m_hA, 0, NULL, NULL);
	V_RETURN_CL(clErr, "Error copying data A from host to device");
	clErr = clEnqueueWriteBuffer(CommandQueue, m_dB, CL_FALSE, 0, m_ArraySize * sizeof(int), m_hB, 0, NULL, NULL);
	V_RETURN_CL(clErr, "Error copying data B from host to device");


	/////////////////////////////////////////
	// Sect. 4.6.
	
	//execute the kernel: one thread for each element!

			// Sect. 4.7.: rewrite the kernel call to use our ProfileKernel()
			//				utility function to measure execution time.
			//				Also print out the execution time.
			
	// Determine number of thread groups and launch kernel

	size_t globalWorkSize = CLUtil::GetGlobalWorkSize(m_ArraySize, LocalWorkSize[0]);
	size_t nGroups = globalWorkSize / LocalWorkSize[0];
	cout << "Executing " << globalWorkSize << " threads in " << nGroups << " groups of size " << LocalWorkSize[0] << endl;

	// standard call but we use Profile Kernel()
	//clErr = clEnqueueNDRangeKernel(CommandQueue, m_Kernel, 1, NULL, &globalWorkSize, LocalWorkSize, 0, NULL, NULL);
	//V_RETURN_CL(clErr, "Error executing kernel");

	CLUtil::ProfileKernel(CommandQueue, m_Kernel, 1, &globalWorkSize, LocalWorkSize, 1000);


	// read back results synchronously.
	//This command has to be blocking, since we need the data
	clErr = clEnqueueReadBuffer(CommandQueue, m_dC, CL_TRUE, 0, m_ArraySize * sizeof(int), m_hGPUResult, 0, NULL, NULL);
	V_RETURN_CL(clErr, "Error reading result array");
}

bool CSimpleArraysTask::ValidateResults()
{
	return (memcmp(m_hC, m_hGPUResult, m_ArraySize * sizeof(float)) == 0);
}

///////////////////////////////////////////////////////////////////////////////
