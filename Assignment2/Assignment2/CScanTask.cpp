/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CScanTask.h"

#include "../Common/CLUtil.h"
#include "../Common/CTimer.h"

#include <string.h>

using namespace std;

// number of banks in the local memory. This can be used to avoid bank conflicts
// but we also need to allocate more local memory for that.
#define NUM_BANKS	32

///////////////////////////////////////////////////////////////////////////////
// CScanTask

// only useful for debug info
const string g_kernelNames[2] = 
{
	"scanNaive",
	"scanWorkEfficient"
};

CScanTask::CScanTask(size_t ArraySize, size_t MinLocalWorkSize)
	: m_N(ArraySize), m_hArray(NULL), m_hResultCPU(NULL), m_hResultGPU(NULL),
	m_dPingArray(NULL), m_dPongArray(NULL),
	m_Program(NULL), 
	m_ScanNaiveKernel(NULL), m_ScanWorkEfficientKernel(NULL), m_ScanWorkEfficientAddKernel(NULL)
{
	// compute the number of levels that we need for the work-efficient algorithm

	m_MinLocalWorkSize = MinLocalWorkSize;

	m_nLevels = 1;
	size_t N = ArraySize;
	while (N > 0){
		N /= 2 * m_MinLocalWorkSize;
		m_nLevels++;
	}

	// Reset validation results
	for (int i = 0; i < (int)ARRAYLEN(m_bValidationResults); i++)
		m_bValidationResults[i] = false;
}

CScanTask::~CScanTask()
{
	ReleaseResources();
}

bool CScanTask::InitResources(cl_device_id Device, cl_context Context)
{
	//CPU resources
	m_hArray	 = new unsigned int[m_N];
	m_hResultCPU = new unsigned int[m_N];
	m_hResultGPU = new unsigned int[m_N];

	//fill the array with some values
	for(unsigned int i = 0; i < m_N; i++)
		//m_hArray[i] = 1;			// Use this for debugging
		m_hArray[i] = rand() & 15;		// TODO remove debuggin

	//device resources
	// ping-pong buffers
	cl_int clError, clError2;
	m_dPingArray = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_N, NULL, &clError2);
	clError = clError2;
	m_dPongArray = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_N, NULL, &clError2);
	clError |= clError2;

	// level buffer
	m_dLevelArrays = new cl_mem[m_nLevels];
	unsigned int N = m_N;
	for (unsigned int i = 0; i < m_nLevels; i++) {
		m_dLevelArrays[i] = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * N, NULL, &clError2);
		clError |= clError2;
		N = max(N / (2 * m_MinLocalWorkSize), m_MinLocalWorkSize);
	}
	V_RETURN_FALSE_CL(clError, "Error allocating device arrays");

	//load and compile kernels
	string programCode;

	CLUtil::LoadProgramSourceToMemory("../Assignment2/Scan.cl", programCode);
	m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
	if(m_Program == nullptr) return false;

	//create kernels
	m_ScanNaiveKernel = clCreateKernel(m_Program, "Scan_Naive", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel.");

	m_ScanWorkEfficientKernel = clCreateKernel(m_Program, "Scan_WorkEfficient", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel.");

	m_ScanWorkEfficientAddKernel = clCreateKernel(m_Program, "Scan_WorkEfficientAdd", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel.");

	return true;
}

void CScanTask::ReleaseResources()
{
	// host resources
	SAFE_DELETE_ARRAY(m_hArray);

	SAFE_DELETE_ARRAY(m_hResultCPU);
	SAFE_DELETE_ARRAY(m_hResultGPU);

	// device resources
	SAFE_RELEASE_MEMOBJECT(m_dPingArray);
	SAFE_RELEASE_MEMOBJECT(m_dPongArray);

	if(m_dLevelArrays)
		for (unsigned int i = 0; i < m_nLevels; i++) {
			SAFE_RELEASE_MEMOBJECT(m_dLevelArrays[i]);
		}
	SAFE_DELETE_ARRAY(m_dLevelArrays);

	SAFE_RELEASE_KERNEL(m_ScanNaiveKernel);
	SAFE_RELEASE_KERNEL(m_ScanWorkEfficientKernel);
	SAFE_RELEASE_KERNEL(m_ScanWorkEfficientAddKernel);

	SAFE_RELEASE_PROGRAM(m_Program);
}

void CScanTask::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	cout << endl;

	ValidateTask(Context, CommandQueue, LocalWorkSize, 0);
	ValidateTask(Context, CommandQueue, LocalWorkSize, 1);

	cout << endl;

	TestPerformance(Context, CommandQueue, LocalWorkSize, 0);
	TestPerformance(Context, CommandQueue, LocalWorkSize, 1);

	cout << endl;
}

void CScanTask::ComputeCPU()
{
	CTimer timer;
	timer.Start();

	unsigned int nIterations = 1;
	for(unsigned int j = 0; j < nIterations; j++) {
		unsigned int sum = 0;
		for(unsigned int i = 0; i < m_N; i++) {
			sum += m_hArray[i];
			m_hResultCPU[i] = sum; 
		}
	}

	timer.Stop();
	double ms = timer.GetElapsedMilliseconds() / double(nIterations);
	cout << "  average time: " << ms << " ms, throughput: " << 1.0e-6 * (double)m_N / ms << " Gelem/s" <<endl;
}

bool CScanTask::ValidateResults()
{
	bool success = true;

	for(int i = 0; i < 2; i++)
		if(!m_bValidationResults[i])
		{
			cout<<"Validation of reduction kernel "<<g_kernelNames[i]<<" failed." << endl;
			success = false;
		}

	return success;
}

void CScanTask::Scan_Naive(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{

	// TO DO: Implement naive version of scan

	// NOTE: make sure that the final result is always in the variable m_dPingArray
	// as this is read back for the correctness check
	// (CReductionTask::ValidateTask)
	//
	// hint: for example, you can use swap(m_dPingArray, m_dPongArray) at the end of your for loop...

	// starting iteration parameters	
	cl_int clError;
	size_t myLocalWorkSize = LocalWorkSize[0];
	size_t globalWorkSize = m_N;							// number of threads in each iteration

	for (uint i = 1; i <= m_N/2; i<<=1)
	{
		// parameters for this iteration
		myLocalWorkSize = globalWorkSize < myLocalWorkSize ? globalWorkSize : LocalWorkSize[0];
		// cout << "GlobalWorkSize: "<<globalWorkSize<<", myLocalWorkSize: "<<myLocalWorkSize<<", nWorkGroups: "<<nWorkGroups<<endl;

		// SET KERNEL ARGUMENTS ///////////////////////////////////////////////////////////////////
		// set first argument: pointer of in array
		clError = clSetKernelArg(m_ScanNaiveKernel, 0, sizeof(cl_mem), (void*) &m_dPingArray);
		// set second argument: pointer of out array
		clError = clSetKernelArg(m_ScanNaiveKernel, 1, sizeof(cl_mem), (void*) &m_dPongArray);
		// set third argument: N
		clError = clSetKernelArg(m_ScanNaiveKernel, 2, sizeof(uint), (void*) &m_N);
		// set third argument: offset
		clError = clSetKernelArg(m_ScanNaiveKernel, 3, sizeof(uint), &i);
		V_RETURN_CL(clError, "Failed to set kernel args: ScanNaive");
		///////////////////////////////////////////////////////////////////////////////////////////

		// RUN KERNEL /////////////////////////////////////////////////////////////////////////////
		clError = clEnqueueNDRangeKernel(CommandQueue, m_ScanNaiveKernel, 1, NULL,
										&globalWorkSize, &myLocalWorkSize,
										0, NULL, NULL);	
		V_RETURN_CL(clError, "Failed to execute Kernel: ScanNaive");
		///////////////////////////////////////////////////////////////////////////////////////////
		// ping pong:
		swap(m_dPingArray, m_dPongArray);
	}
	// ping is the last output array, as they are being swapped at the end of each iteration

}

void CScanTask::Scan_WorkEfficient(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	cl_int clError;
	size_t myLocalWorkSize = LocalWorkSize[0];
	size_t globalWorkSize = m_N / 2;

	// cout << "Running Scan in " << m_nLevels << " levels, array size: " << m_N << endl;
	
	// this code works for local PPS
	/*
	// SET KERNEL ARGUMENTS ///////////////////////////////////////////////////////////////////
	// set first argument: pointer of in array
	clError = clSetKernelArg(m_ScanWorkEfficientKernel, 0, sizeof(cl_mem), (void*) &m_dLevelArrays[0]);
	// set second argument: pointer of out higherLevelArray
	// unsigned int k = m_nLevels > 1 ? 1 : 0;		// pass back to level 0 if local PPS is enough
	clError = clSetKernelArg(m_ScanWorkEfficientKernel, 1, sizeof(cl_mem), (void*) &m_dLevelArrays[1]);
	// set third argument: local block
	size_t size = 2 * myLocalWorkSize;		// number of elements to store
	size += size/NUM_BANKS;					// number of pads
	// cout << "LocalBlock size: " << size << endl;
	clError = clSetKernelArg(m_ScanWorkEfficientKernel, 2, size * sizeof(uint), NULL);
	V_RETURN_CL(clError, "Failed to set kernel args: ScanWorkEfficient");
	///////////////////////////////////////////////////////////////////////////////////////////

	// cout << "GlobalWorkSize: "<<globalWorkSize<<", myLocalWorkSize: "<<myLocalWorkSize<<endl;
	// RUN KERNEL /////////////////////////////////////////////////////////////////////////////
	clError = clEnqueueNDRangeKernel(CommandQueue, m_ScanWorkEfficientKernel, 1, NULL,
									&globalWorkSize, &myLocalWorkSize,
									0, NULL, NULL);	
	V_RETURN_CL(clError, "Failed to execute Kernel: ScanWorkEfficient (Local PPS)");
	///////////////////////////////////////////////////////////////////////////////////////////
	return;
	// Make sure that the local prefix sum works before you start experimenting with large arrays
	// works fine :-)
	*/
	
	// level -1 is just a output for level -2
	uint i;		// levels we write to
	for (i = 0; i < m_nLevels - 1; i++) {
		// SET KERNEL ARGUMENTS ///////////////////////////////////////////////////////////////////
		// set first argument: pointer of in array
		clError = clSetKernelArg(m_ScanWorkEfficientKernel, 0, sizeof(cl_mem), (void*) &m_dLevelArrays[i]);
		// set second argument: pointer of out higherLevelArray
		// unsigned int k = m_nLevels == i+1 ? i : i+1;		// pass back to level 0 if local PPS is enough
		// unsigned int k = min(m_nLevels-1, i+1);			// dont go out of bounds
		// if i = k: just perform a local PPS. This is the top level
		clError = clSetKernelArg(m_ScanWorkEfficientKernel, 1, sizeof(cl_mem), (void*) &m_dLevelArrays[i+1]);
		// set third argument: local block
		size_t size = 2 * myLocalWorkSize;		// number of elements to store
		size += size/NUM_BANKS;					// number of pads
		// cout << "LocalBlock size: " << size << endl;
		clError = clSetKernelArg(m_ScanWorkEfficientKernel, 2, size * sizeof(uint), NULL);
		V_RETURN_CL(clError, "Failed to set kernel args: ScanWorkEfficient (Group PPS)");
		///////////////////////////////////////////////////////////////////////////////////////////

		if (myLocalWorkSize == 0)		// this can happen if globalWorkSize is 1 
			myLocalWorkSize = 1;
		// cout << endl << "GROUP PPS: GlobalWorkSize: "<<globalWorkSize<<", myLocalWorkSize: "<<myLocalWorkSize<<endl;
		// RUN KERNEL /////////////////////////////////////////////////////////////////////////////
		clError = clEnqueueNDRangeKernel(CommandQueue, m_ScanWorkEfficientKernel, 1, NULL,
										&globalWorkSize, &myLocalWorkSize,
										0, NULL, NULL);	
		V_RETURN_CL(clError, "Failed to execute Kernel: ScanWorkEfficient (Group PPS)");
		///////////////////////////////////////////////////////////////////////////////////////////

		if (globalWorkSize / myLocalWorkSize / 2 <= 1) {
			// only one result is being written, no PPS needed
			i++;
			break;
		}

		// set parameters for the next step
		globalWorkSize /= myLocalWorkSize * 2;
		if (globalWorkSize < myLocalWorkSize) {
			myLocalWorkSize = globalWorkSize;
		} 
	}
	
	// now distribute all group prefixes to the elements
	for (; i>0; i--) {		// reuse the count from above
		// SET KERNEL ARGUMENTS ///////////////////////////////////////////////////////////////////
		// set first argument: pointer of higherLevelArray
		clError = clSetKernelArg(m_ScanWorkEfficientAddKernel, 0, sizeof(cl_mem), (void*) &m_dLevelArrays[i]);
		// set second argument: pointer of out array 
		clError = clSetKernelArg(m_ScanWorkEfficientAddKernel, 1, sizeof(cl_mem), (void*) &m_dLevelArrays[i-1]);
		// set third argument: local block
		// size_t size = 2 * myLocalWorkSize;		// number of elements to store
		// size += size/NUM_BANKS;					// number of pads
		// cout << "LocalBlock size: " << size << endl;
		// no need for a local block
		clError = clSetKernelArg(m_ScanWorkEfficientAddKernel, 2, 1, NULL);
		V_RETURN_CL(clError, "Failed to set kernel args: ScanWorkEfficientAdd");
		///////////////////////////////////////////////////////////////////////////////////////////

		if (myLocalWorkSize == 0)		// this can happen if globalWorkSize is 1 
			myLocalWorkSize = 1;
		// cout << endl << "ADD: GlobalWorkSize: "<<globalWorkSize<<", myLocalWorkSize: "<<myLocalWorkSize<<endl;
		// RUN KERNEL /////////////////////////////////////////////////////////////////////////////
		clError = clEnqueueNDRangeKernel(CommandQueue, m_ScanWorkEfficientAddKernel, 1, NULL,
										&globalWorkSize, &myLocalWorkSize,
										0, NULL, NULL);	
		V_RETURN_CL(clError, "Failed to execute Kernel: ScanWorkEfficientAdd");
		///////////////////////////////////////////////////////////////////////////////////////////
		myLocalWorkSize = LocalWorkSize[0];
		globalWorkSize *= 2 * LocalWorkSize[0];
	}


}

void CScanTask::ValidateTask(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int Task)
{
	//run selected task
	switch (Task){
		case 0:
			V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N * sizeof(cl_uint), m_hArray, 0, NULL, NULL), "Error copying data from host to device!");
			Scan_Naive(Context, CommandQueue, LocalWorkSize);
			V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_dPingArray, CL_TRUE, 0, m_N * sizeof(cl_uint), m_hResultGPU, 0, NULL, NULL), "Error reading data from device!");
			break;
		case 1:
			V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dLevelArrays[0], CL_FALSE, 0, m_N * sizeof(cl_uint), m_hArray, 0, NULL, NULL), "Error copying data from host to device!");
			Scan_WorkEfficient(Context, CommandQueue, LocalWorkSize);
			V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_dLevelArrays[0], CL_TRUE, 0, m_N * sizeof(cl_uint), m_hResultGPU, 0, NULL, NULL), "Error reading data from device!");
			break;
	}

	// validate results
	// for (uint i = 0; i<2*LocalWorkSize[0]; i++) cout << m_hResultGPU[i] << ", ";		// debug
	m_bValidationResults[Task] =( memcmp(m_hResultCPU, m_hResultGPU, m_N * sizeof(unsigned int)) == 0);
}

void CScanTask::TestPerformance(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int Task)
{
	cout << "Testing performance of task " << g_kernelNames[Task] << endl;

	//write input data to the GPU
	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N * sizeof(cl_uint), m_hArray, 0, NULL, NULL), "Error copying data from host to device!");
	//finish all before we start meassuring the time
	V_RETURN_CL(clFinish(CommandQueue), "Error finishing the queue!");

	CTimer timer;
	timer.Start();

	//run the kernel N times
	unsigned int nIterations = 100;
	for(unsigned int i = 0; i < nIterations; i++) {
		//run selected task
		switch (Task){
			case 0:
				Scan_Naive(Context, CommandQueue, LocalWorkSize);
				break;
			case 1:
				Scan_WorkEfficient(Context, CommandQueue, LocalWorkSize);
				break;
		}
	}

	//wait until the command queue is empty again
	V_RETURN_CL(clFinish(CommandQueue), "Error finishing the queue!");

	timer.Stop();

	double ms = timer.GetElapsedMilliseconds() / double(nIterations);
	cout << "  average time: " << ms << " ms, throughput: " << 1.0e-6 * (double)m_N / ms << " Gelem/s" <<endl;
}



///////////////////////////////////////////////////////////////////////////////
