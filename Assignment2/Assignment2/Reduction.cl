
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_InterleavedAddressing(__global uint* array, uint stride) 
{
	int x = 2 * stride * get_global_id(0);
	//if (x + stride > 2*8388608){printf("%i\n", x);} else	// debug
	array[ x ] = array[ x ] + array[ x + stride ];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_SequentialAddressing(__global uint* array, uint stride) 
{
	int x = get_global_id(0);
	array[ x ] = array[ x ] + array [ x + get_global_size(0)];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_Decomp(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	int LID = get_local_id(0);
	int GID = get_global_id(0);
	int numOfThreads = get_local_size(0);

	// First step: Load Data into localBlock while reducing it already
	localBlock[ LID ] = inArray[ GID ] + inArray[ GID + get_global_size(0) ];
	barrier(CLK_LOCAL_MEM_FENCE);	// wait until array is written

	// second part: reduce localBlock
	// implementation is like sequential addressing
	for (int i = 1; numOfThreads > 0; i++) {
		numOfThreads = numOfThreads >> 1;	// half the number of threads in each step
		if (LID < numOfThreads) {			// workItem used for reduction
			localBlock[ LID ] += localBlock[ LID + numOfThreads];
		}
		barrier(CLK_LOCAL_MEM_FENCE);	// wait until reduction step is complete
	}

	// write back
	if (LID == 0) { // only one thread writes back
		outArray[ get_group_id(0) ] = localBlock[ 0 ];
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompUnroll(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	int LID = get_local_id(0);
	int GID = get_global_id(0);
	int numOfThreads = get_local_size(0);

	// First step: Load Data into localBlock while reducing it already
	localBlock[ LID ] = inArray[ GID ] + inArray[ GID + get_global_size(0) ];
	barrier(CLK_LOCAL_MEM_FENCE);	// wait until array is written

	// second part: reduce localBlock
	// implementation is like sequential addressing
	int stride;
	for (int i = 1; (numOfThreads >> i) > 32; i++) {
		stride = numOfThreads >> i;	// half the number of threads in each step
		if (LID < stride) {			// workItem used for reduction
			localBlock[ LID ] += localBlock[ LID + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);	// wait until reduction step is complete
	}
	
	int max_stride = numOfThreads / 2;
	// unroll the loop
	if (LID < 32 && 32 <= max_stride) localBlock[ LID ] += localBlock[ LID + 32];
	if (LID < 16 && 16 <= max_stride) localBlock[ LID ] += localBlock[ LID + 16];
	if (LID <  8 &&  8 <= max_stride) localBlock[ LID ] += localBlock[ LID +  8];
	if (LID <  4 &&  4 <= max_stride) localBlock[ LID ] += localBlock[ LID +  4];
	if (LID <  2 &&  2 <= max_stride) localBlock[ LID ] += localBlock[ LID +  2];
	if (LID <  1 &&  1 <= max_stride) localBlock[ LID ] += localBlock[ LID +  1];


	// write back
	if (LID == 0) { // only one thread writes back
		outArray[ get_group_id(0) ] = localBlock[ 0 ];
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompAtomics(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localSum)
{
	int GID = get_global_id(0);
	int LID = get_local_id(0);
	// initialize localSum
	if (LID == 0) *localSum = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_add(localSum, inArray[GID] + inArray[GID + get_global_size(0)]);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (LID == 0) outArray[get_group_id(0)] = *localSum;
}

__kernel void Reduction_LoadMax(const __global uint* inArray, __global uint* outArray, uint maxElements, __local uint* localSum)
{
	int local_size = get_local_size(0);
	int LID = get_local_id(0);
	int elementsPerWorkItem = maxElements / local_size;

	// initialize localSum
	if (LID == 0) *localSum = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	uint workItemSum = 0;
	//if (0 == (LID|get_group_id(0))) printf("ePWI: %i\n", elementsPerWorkItem);
	for (int i = 0; i < elementsPerWorkItem; i++)
		workItemSum += inArray[LID + i*elementsPerWorkItem];

	atomic_add(localSum, workItemSum);
	barrier(CLK_LOCAL_MEM_FENCE);
	// store data:
	if (LID == 0) outArray[get_group_id(0)] = *localSum;

}