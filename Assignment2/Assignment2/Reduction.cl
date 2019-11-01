
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
	// TO DO: Kernel implementation
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompAtomics(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localSum)
{
	// TO DO: Kernel implementation
}
