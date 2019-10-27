/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CAssignment1.h"

#include "CSimpleArraysTask.h"
#include "CMatrixRotateTask.h"

#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CAssignment1

bool CAssignment1::DoCompute()
{
	// Task 1: simple array addition.
	cout << "Running vector addition example..." << endl << endl;
	size_t worksizes[4] = {16,64,256,1024};
	size_t vectorsizes[4] = {1<<11, 1<<15, 1<<20, 1<<25};
	for (int i = 0; i < 4; i++) {
		for (size_t j = 0; j < 4; j++) {
			size_t localWorkSize[3] = { worksizes[j], 1, 1 };
			CSimpleArraysTask task(vectorsizes[i]);
			RunComputeTask(task, localWorkSize);
		}

	}
	
	if (false){
		// skip those
	{
		size_t localWorkSize[3] = { 1, 1, 1 };
		CSimpleArraysTask task(1048576);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {256, 1, 1};
		CSimpleArraysTask task(1048576);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t LocalWorkSize[3] = {512, 1, 1};
		CSimpleArraysTask task(1048576);
		RunComputeTask(task, LocalWorkSize);
	}
	}

	// Task 2: matrix rotation.
	std::cout << std::endl << std::endl << "Running matrix rotation example..." << std::endl << std::endl;
	{
		size_t LocalWorkSize[3] = {32, 16, 1};
		//CMatrixRotateTask task(2048, 1025);
		CMatrixRotateTask task(2048, 1024);
		//CMatrixRotateTask task(32, 32);
		RunComputeTask(task, LocalWorkSize);
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////
