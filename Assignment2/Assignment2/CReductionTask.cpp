/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CReductionTask.h"

#include "../Common/CLUtil.h"
#include "../Common/CTimer.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CReductionTask

string g_kernelNames[6] = {
	"interleavedAddressing",
	"sequentialAddressing",
	"kernelDecomposition",
	"kernelDecompositionUnroll",
	"kernelDecompositionAtomics",
	"kernelLoadMax"
};

CReductionTask::CReductionTask(size_t ArraySize)
	: m_N(ArraySize), m_hInput(NULL), 
	m_dPingArray(NULL),
	m_dPongArray(NULL),
	m_Program(NULL), 
	m_InterleavedAddressingKernel(NULL), m_SequentialAddressingKernel(NULL), m_DecompKernel(NULL), m_DecompUnrollKernel(NULL), m_DecompAtomicsKernel(NULL)
{
}

CReductionTask::~CReductionTask()
{
	ReleaseResources();
}

bool CReductionTask::InitResources(cl_device_id Device, cl_context Context)
{
	//CPU resources
	m_hInput = new unsigned int[m_N];

	//fill the array with some values
	for(unsigned int i = 0; i < m_N; i++) 
		//m_hInput[i] = 1;			// Use this for debugging
		m_hInput[i] = rand() & 15;	// TODO remove debugging

	//device resources
	cl_int clError, clError2;
	m_dPingArray = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_N, NULL, &clError2);
	clError = clError2;
	m_dPongArray = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_N, NULL, &clError2);
	clError |= clError2;
	V_RETURN_FALSE_CL(clError, "Error allocating device arrays");

	//load and compile kernels
	string programCode;

	CLUtil::LoadProgramSourceToMemory("../Assignment2/Reduction.cl", programCode);
	m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
	if(m_Program == nullptr) return false;

	//create kernels
	m_InterleavedAddressingKernel = clCreateKernel(m_Program, "Reduction_InterleavedAddressing", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: Reduction_InterleavedAddressing.");

	m_SequentialAddressingKernel = clCreateKernel(m_Program, "Reduction_SequentialAddressing", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: Reduction_SequentialAddressing.");

	m_DecompKernel = clCreateKernel(m_Program, "Reduction_Decomp", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: Reduction_Decomp.");

	m_DecompUnrollKernel = clCreateKernel(m_Program, "Reduction_DecompUnroll", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: Reduction_DecompUnroll.");

	m_DecompAtomicsKernel = clCreateKernel(m_Program, "Reduction_DecompAtomics", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: Reduction_DecompAtomics.");

	m_LoadMaxKernel = clCreateKernel(m_Program, "Reduction_LoadMax", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel: Reduction_LoadMax.");

	return true;
}

void CReductionTask::ReleaseResources()
{
	// host resources
	SAFE_DELETE_ARRAY(m_hInput);

	// device resources
	SAFE_RELEASE_MEMOBJECT(m_dPingArray);
	SAFE_RELEASE_MEMOBJECT(m_dPongArray);

	SAFE_RELEASE_KERNEL(m_InterleavedAddressingKernel);
	SAFE_RELEASE_KERNEL(m_SequentialAddressingKernel);
	SAFE_RELEASE_KERNEL(m_DecompKernel);
	SAFE_RELEASE_KERNEL(m_DecompUnrollKernel);
	SAFE_RELEASE_KERNEL(m_DecompAtomicsKernel);
	SAFE_RELEASE_KERNEL(m_LoadMaxKernel);

	SAFE_RELEASE_PROGRAM(m_Program);
}

void CReductionTask::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	ExecuteTask(Context, CommandQueue, LocalWorkSize, 0);
	ExecuteTask(Context, CommandQueue, LocalWorkSize, 1);
	ExecuteTask(Context, CommandQueue, LocalWorkSize, 2);
	ExecuteTask(Context, CommandQueue, LocalWorkSize, 3);
	ExecuteTask(Context, CommandQueue, LocalWorkSize, 4);
	ExecuteTask(Context, CommandQueue, LocalWorkSize, 5);

	TestPerformance(Context, CommandQueue, LocalWorkSize, 0);
	TestPerformance(Context, CommandQueue, LocalWorkSize, 1);
	TestPerformance(Context, CommandQueue, LocalWorkSize, 2);
	TestPerformance(Context, CommandQueue, LocalWorkSize, 3);
	TestPerformance(Context, CommandQueue, LocalWorkSize, 4);
	TestPerformance(Context, CommandQueue, LocalWorkSize, 5);

}

void CReductionTask::ComputeCPU()
{
	CTimer timer;
	timer.Start();

	unsigned int nIterations = 10;
	for(unsigned int j = 0; j < nIterations; j++) {
		m_resultCPU = m_hInput[0];
		for(unsigned int i = 1; i < m_N; i++) {
			m_resultCPU += m_hInput[i]; 
		}
	}

	timer.Stop();

	double ms = timer.GetElapsedMilliseconds() / double(nIterations);
	cout << "  average time: " << ms << " ms, throughput: " << 1.0e-6 * (double)m_N / ms << " Gelem/s" <<endl;
	// cout << "CPU result: " << m_resultCPU << endl;		// debug
}

bool CReductionTask::ValidateResults()
{
	bool success = true;

	for(int i = 0; i < 6; i++)
		if(m_resultGPU[i] != m_resultCPU)
		{
			cout<<"Validation of reduction kernel "<<g_kernelNames[i]<<" failed." << endl;
			success = false;
		}

	return success;
}

void CReductionTask::Reduction_InterleavedAddressing(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	// log can be computed by right-shifting
	cl_int clError;
	size_t myLocalWorkSize = LocalWorkSize[0];
	for (size_t i = 1; m_N >> i > 0; i++) {
		// set first argument: pointer of array
		clError = clSetKernelArg(m_InterleavedAddressingKernel, 0, sizeof(cl_mem), (void*) &m_dPingArray);
		// set second argument: stride
		uint stride = 1 << (i-1);
		clError |= clSetKernelArg(m_InterleavedAddressingKernel, 1, sizeof(int), &stride);
		V_RETURN_CL(clError, "Failed to set kernel args: InterleavedAddressing");

		size_t globalWorkSize = (size_t) (m_N >> i);		// number of threads

		// adapt LocalWorkSize to the Problem
		myLocalWorkSize = globalWorkSize < myLocalWorkSize ? globalWorkSize : LocalWorkSize[0];
		//cout << "GlobalSize: " << globalWorkSize << ", LocalWorksize: " << myLocalWorkSize << endl;
		clError = clEnqueueNDRangeKernel(CommandQueue, m_InterleavedAddressingKernel, 1, NULL,
										&globalWorkSize, &myLocalWorkSize,
										0, NULL, NULL);	
		V_RETURN_CL(clError, "Failed to execute Kernel: InterleavedAddressing");							
	}
}

void CReductionTask::Reduction_SequentialAddressing(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	// log can be computed by right-shifting
	cl_int clError;
	size_t myLocalWorkSize = LocalWorkSize[0];
	for (size_t i = 1; m_N >> i > 0; i++) {
		// set first argument: pointer of array
		clError = clSetKernelArg(m_SequentialAddressingKernel, 0, sizeof(cl_mem), (void*) &m_dPingArray);
		// set second argument: stride
		uint stride = 1 << (i-1);
		clError |= clSetKernelArg(m_SequentialAddressingKernel, 1, sizeof(int), &stride);
		V_RETURN_CL(clError, "Failed to set kernel args: SequentialAddressing");

		size_t globalWorkSize = (size_t) (m_N >> i);		// number of threads

		// adapt LocalWorkSize to the Problem
		myLocalWorkSize = globalWorkSize < myLocalWorkSize ? globalWorkSize : LocalWorkSize[0];
		//cout << "GlobalSize: " << globalWorkSize << ", LocalWorksize: " << myLocalWorkSize << endl;
		clError = clEnqueueNDRangeKernel(CommandQueue, m_SequentialAddressingKernel, 1, NULL,
										&globalWorkSize, &myLocalWorkSize,
										0, NULL, NULL);	
		V_RETURN_CL(clError, "Failed to execute Kernel: SequentialAddressing");
	}
}

void CReductionTask::Reduction_Decomp(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	// starting iteration parameters	
	cl_int clError;
	size_t myLocalWorkSize = LocalWorkSize[0];
	int nWorkGroups = m_N; 							// this equals the number of to be reduced elements in the next step
	size_t globalWorkSize;							// number of threads in each iteration

	do
	{
		// parameters for this iteration
		globalWorkSize = nWorkGroups / 2;
		myLocalWorkSize = globalWorkSize < myLocalWorkSize ? globalWorkSize : LocalWorkSize[0];
		nWorkGroups = globalWorkSize / myLocalWorkSize;
		// cout << "GlobalWorkSize: "<<globalWorkSize<<", myLocalWorkSize: "<<myLocalWorkSize<<", nWorkGroups: "<<nWorkGroups<<endl;

		// SET KERNEL ARGUMENTS ///////////////////////////////////////////////////////////////////
		// set first argument: pointer of in array
		clError = clSetKernelArg(m_DecompKernel, 0, sizeof(cl_mem), (void*) &m_dPingArray);
		// set second argument: pointer of out array
		clError = clSetKernelArg(m_DecompKernel, 1, sizeof(cl_mem), (void*) &m_dPongArray);
		// set third argument: N
		clError = clSetKernelArg(m_DecompKernel, 2, sizeof(uint), (void*) &m_N);
		// set third argument: pointer of localBlock
		clError = clSetKernelArg(m_DecompKernel, 3, myLocalWorkSize * sizeof(uint), NULL);
		V_RETURN_CL(clError, "Failed to set kernel args: Decomp");
		///////////////////////////////////////////////////////////////////////////////////////////

		// RUN KERNEL /////////////////////////////////////////////////////////////////////////////
		clError = clEnqueueNDRangeKernel(CommandQueue, m_DecompKernel, 1, NULL,
										&globalWorkSize, &myLocalWorkSize,
										0, NULL, NULL);	
		V_RETURN_CL(clError, "Failed to execute Kernel: Decomp");
		///////////////////////////////////////////////////////////////////////////////////////////
		// ping pong:
		swap(m_dPingArray, m_dPongArray);
	} while (nWorkGroups != 1);
	// ping is the last output array, as they are being swapped at the end of each iteration
}

void CReductionTask::Reduction_DecompUnroll(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	// starting iteration parameters	
	cl_int clError;
	size_t myLocalWorkSize = LocalWorkSize[0];
	int nWorkGroups = m_N; 							// this equals the number of to be reduced elements in the next step
	size_t globalWorkSize;							// number of threads in each iteration

	do
	{
		// parameters for this iteration
		globalWorkSize = nWorkGroups / 2;
		myLocalWorkSize = globalWorkSize < myLocalWorkSize ? globalWorkSize : LocalWorkSize[0];
		nWorkGroups = globalWorkSize / myLocalWorkSize;
		// cout << "GlobalWorkSize: "<<globalWorkSize<<", myLocalWorkSize: "<<myLocalWorkSize<<", nWorkGroups: "<<nWorkGroups<<endl;

		// SET KERNEL ARGUMENTS ///////////////////////////////////////////////////////////////////
		// set first argument: pointer of in array
		clError = clSetKernelArg(m_DecompUnrollKernel, 0, sizeof(cl_mem), (void*) &m_dPingArray);
		// set second argument: pointer of out array
		clError = clSetKernelArg(m_DecompUnrollKernel, 1, sizeof(cl_mem), (void*) &m_dPongArray);
		// set third argument: N
		uint n = myLocalWorkSize;
		clError = clSetKernelArg(m_DecompUnrollKernel, 2, sizeof(uint), (void*) &n);
		// set forth argument: pointer of localBlock
		clError = clSetKernelArg(m_DecompUnrollKernel, 3, myLocalWorkSize * sizeof(uint), NULL);
		V_RETURN_CL(clError, "Failed to set kernel args: DecompUnroll");
		///////////////////////////////////////////////////////////////////////////////////////////

		// RUN KERNEL /////////////////////////////////////////////////////////////////////////////
		clError = clEnqueueNDRangeKernel(CommandQueue, m_DecompUnrollKernel, 1, NULL,
										&globalWorkSize, &myLocalWorkSize,
										0, NULL, NULL);	
		V_RETURN_CL(clError, "Failed to execute Kernel: DecompUnroll");
		///////////////////////////////////////////////////////////////////////////////////////////
		// ping pong:
		swap(m_dPingArray, m_dPongArray);
	} while (nWorkGroups != 1);
	// ping is the last output array, as they are being swapped at the end of each iteration
}

void CReductionTask::Reduction_DecompAtomics(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{

	// TO DO: Implement reduction with Atomics

	cl_int clError;
	size_t myLocalWorkSize = LocalWorkSize[0];
	int nWorkGroups = m_N; 							// this equals the number of to be reduced elements in the next step
	size_t globalWorkSize;							// number of threads in each iteration

	do
	{
		// parameters for this iteration
		globalWorkSize = nWorkGroups / 2;
		myLocalWorkSize = globalWorkSize < myLocalWorkSize ? globalWorkSize : LocalWorkSize[0];
		nWorkGroups = globalWorkSize / myLocalWorkSize;
		// cout << "GlobalWorkSize: "<<globalWorkSize<<", myLocalWorkSize: "<<myLocalWorkSize<<", nWorkGroups: "<<nWorkGroups<<endl;

		// SET KERNEL ARGUMENTS ///////////////////////////////////////////////////////////////////
		// set first argument: pointer of in array
		clError = clSetKernelArg(m_DecompAtomicsKernel, 0, sizeof(cl_mem), (void*) &m_dPingArray);
		// set second argument: pointer of out array
		clError = clSetKernelArg(m_DecompAtomicsKernel, 1, sizeof(cl_mem), (void*) &m_dPongArray);
		// set third argument: N
		uint n = myLocalWorkSize;
		clError = clSetKernelArg(m_DecompAtomicsKernel, 2, sizeof(uint), (void*) &n);
		// set forth argument: localSum
		clError = clSetKernelArg(m_DecompAtomicsKernel, 3, sizeof(uint), NULL);
		V_RETURN_CL(clError, "Failed to set kernel args: DecompAtomics");
		///////////////////////////////////////////////////////////////////////////////////////////

		// RUN KERNEL /////////////////////////////////////////////////////////////////////////////
		clError = clEnqueueNDRangeKernel(CommandQueue, m_DecompAtomicsKernel, 1, NULL,
										&globalWorkSize, &myLocalWorkSize,
										0, NULL, NULL);	
		V_RETURN_CL(clError, "Failed to execute Kernel: DecompAtomics");
		///////////////////////////////////////////////////////////////////////////////////////////
		// ping pong:
		swap(m_dPingArray, m_dPongArray);
	} while (nWorkGroups != 1);
	// ping is the last output array, as they are being swapped at the end of each iteration
}

void CReductionTask::Reduction_LoadMax(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	/* 
	Idea of this approach:
	Load as many values as possible into the local memory.
	Then each work-item reduces its fair share of values.

	After that each workitem writes its result with an atomic add to a local value (local sum)

	In the end one work-item of each work-group writes this local sum to the outArray
	
	These steps are iterated
	*/
	// cl_ulong localMemorySize;
	// clGetDeviceIDs(...)
	// clGetDeviceInfo(m_CLDevice, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemorySize, &bufferSize);
	int localMemorySize = 49152;	// in byte
	localMemorySize = 32768;		// better use this: a whole power of 2

	localMemorySize /= sizeof(uint);	// in number of uint

	// cout << "Local memory fits " << localMemorySize << " uint's" << endl;

	cl_int clError;
	size_t myLocalWorkSize = LocalWorkSize[0];
	int nToReduce = m_N; 		// this equals the number of to be reduced elements in the next step
	size_t globalWorkSize;							// number of threads in each iteration

	do
	{
		// parameters for this iteration
		int nGroups = nToReduce / localMemorySize;
		globalWorkSize = nGroups > 0 ? nGroups * myLocalWorkSize : min(nToReduce,(int)myLocalWorkSize);		// reset if too small globalWorkSize
		myLocalWorkSize = globalWorkSize < myLocalWorkSize ? globalWorkSize : LocalWorkSize[0];
		// cout << "GlobalWorkSize: "<<globalWorkSize<<", myLocalWorkSize: "<<myLocalWorkSize<<", nWorkGroups: "<<globalWorkSize/myLocalWorkSize<<endl;

		// SET KERNEL ARGUMENTS ///////////////////////////////////////////////////////////////////
		// set first argument: pointer of in array
		clError = clSetKernelArg(m_LoadMaxKernel, 0, sizeof(cl_mem), (void*) &m_dPingArray);
		// set second argument: pointer of out array
		clError = clSetKernelArg(m_LoadMaxKernel, 1, sizeof(cl_mem), (void*) &m_dPongArray);
		// set third argument: N
		uint maxElements = min(nToReduce, localMemorySize);
		clError = clSetKernelArg(m_LoadMaxKernel, 2, sizeof(uint), (void*) &maxElements);
		// set forth argument: localSum
		clError = clSetKernelArg(m_LoadMaxKernel, 3, sizeof(uint), NULL);
		V_RETURN_CL(clError, "Failed to set kernel args: LoadMax");
		///////////////////////////////////////////////////////////////////////////////////////////

		// RUN KERNEL /////////////////////////////////////////////////////////////////////////////
		clError = clEnqueueNDRangeKernel(CommandQueue, m_LoadMaxKernel, 1, NULL,
										&globalWorkSize, &myLocalWorkSize,
										0, NULL, NULL);	
		V_RETURN_CL(clError, "Failed to execute Kernel: LoadMax");
		///////////////////////////////////////////////////////////////////////////////////////////
		// ping pong:
		swap(m_dPingArray, m_dPongArray);
		
		nToReduce /= localMemorySize;		// number of outputs = number of to be reduced elements in the next step
	} while (nToReduce >= 1);
	// ping is the last output array, as they are being swapped at the end of each iteration
}

void CReductionTask::ExecuteTask(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int Task)
{
	//write input data to the GPU
	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N * sizeof(cl_uint), m_hInput, 0, NULL, NULL), "Error copying data from host to device!");

	//run selected task
	switch (Task){
		case 0:
			Reduction_InterleavedAddressing(Context, CommandQueue, LocalWorkSize);
			break;
		case 1:
			Reduction_SequentialAddressing(Context, CommandQueue, LocalWorkSize);
			break;
		case 2:
			Reduction_Decomp(Context, CommandQueue, LocalWorkSize);
			break;
		case 3:
			Reduction_DecompUnroll(Context, CommandQueue, LocalWorkSize);
			break;
		case 4:
			Reduction_DecompAtomics(Context, CommandQueue, LocalWorkSize);
			break;
		case 5:
			Reduction_LoadMax(Context, CommandQueue, LocalWorkSize);
			break;

	}

	//read back the results synchronously.
	m_resultGPU[Task] = 0;
	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_dPingArray, CL_TRUE, 0, 1 * sizeof(cl_uint), &m_resultGPU[Task], 0, NULL, NULL), "Error reading data from device!");
	// cout << "Result: " << m_resultGPU[Task] << endl; 	// debug
	
}

void CReductionTask::TestPerformance(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int Task)
{
	cout << "Testing performance of task " << g_kernelNames[Task] << endl;

	//write input data to the GPU
	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N * sizeof(cl_uint), m_hInput, 0, NULL, NULL), "Error copying data from host to device!");
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
				Reduction_InterleavedAddressing(Context, CommandQueue, LocalWorkSize);
				break;
			case 1:
				Reduction_SequentialAddressing(Context, CommandQueue, LocalWorkSize);
				break;
			case 2:
				Reduction_Decomp(Context, CommandQueue, LocalWorkSize);
				break;
			case 3:
				Reduction_DecompUnroll(Context, CommandQueue, LocalWorkSize);
				break;
			case 4:
				Reduction_DecompAtomics(Context, CommandQueue, LocalWorkSize);
				break;
			case 5:
				Reduction_LoadMax(Context, CommandQueue, LocalWorkSize);
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
