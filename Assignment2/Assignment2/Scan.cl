


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_Naive(const __global uint* inArray, __global uint* outArray, uint N, uint offset) 
{
	int LID = get_local_id(0);
	int GID = get_global_id(0);
	int numOfThreads = get_local_size(0);
	if (GID < offset)
		outArray[ GID ] = inArray[ GID ];
	else
		outArray[ GID ] = inArray[ GID ] + inArray[ GID - offset ];
}



// Why did we not have conflicts in the Reduction? Because of the sequential addressing (here we use interleaved => we have conflicts).

#define UNROLL
#define NUM_BANKS			32
#define NUM_BANKS_LOG		5
#define SIMD_GROUP_SIZE		32

// Bank conflicts
#define AVOID_BANK_CONFLICTS
#ifdef AVOID_BANK_CONFLICTS
	// TO DO: define your conflict-free macro here
	#define OFFSET(A) (((A)/NUM_BANKS + (A)))
#else
	#define OFFSET(A) (A)
#endif

#define INDEXFROMRIGHT(i) ((2*local_size - 1 - (i)))

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficient(__global uint* array, __global uint* higherLevelArray, __local uint* localBlock) 
{
	// TO DO: Kernel implementation
	int GID = get_global_id(0);
	int LID = get_local_id(0);
	int local_size = get_local_size(0);
	int subArrayStart = get_group_id(0) * 2 * local_size;

	//////////////////////////////////////////////////////
	// init: copy array to local memory	//////////////////
	//////////////////////////////////////////////////////
	localBlock[OFFSET(LID)] = array[subArrayStart + LID];
	localBlock[OFFSET(local_size + LID)] = array[subArrayStart + local_size + LID];

	//////////////////////////////////////////////////////
	// first part: up-sweep	//////////////////////////////
	//////////////////////////////////////////////////////
	int stride = 1;
	do {
		barrier(CLK_LOCAL_MEM_FENCE);
		if (LID < local_size/stride)
			localBlock[OFFSET(INDEXFROMRIGHT(2*stride*LID))] += 
				localBlock[OFFSET(INDEXFROMRIGHT(2*stride*LID + stride))];

		// if (LID==0) printf("current last value: %i\n",localBlock[OFFSET(INDEXFROMRIGHT(2*stride*LID))]);

		// update values vor next iteration
		stride *= 2;
	} while (stride < local_size);

	// set last element 0 ////////////////////////////////
	if (LID == 0) localBlock[OFFSET(INDEXFROMRIGHT(0))] = 0;
	//////////////////////////////////////////////////////
	// second part: down-sweep	//////////////////////////
	//////////////////////////////////////////////////////
	// use stride from up-sweep
	do {
		barrier(CLK_LOCAL_MEM_FENCE);
		if (LID < local_size/stride) {
			// debug 
			/*
			if (LID==0) printf("before: l=%i, r=%i\n",
				localBlock[OFFSET(INDEXFROMRIGHT(2*stride*LID + stride))],
				localBlock[OFFSET(INDEXFROMRIGHT(2*stride*LID))]);
			*/

			// right child = sum of the nodes
			localBlock[OFFSET(INDEXFROMRIGHT(2*stride*LID))] +=
				localBlock[OFFSET(INDEXFROMRIGHT(2*stride*LID + stride))];

			// left child
			localBlock[OFFSET(INDEXFROMRIGHT(2*stride*LID + stride))] = 
				localBlock[OFFSET(INDEXFROMRIGHT(2*stride*LID))] - 
				localBlock[OFFSET(INDEXFROMRIGHT(2*stride*LID + stride))];
		}

		// if (LID==0) printf("current last value: %i\n",localBlock[OFFSET(INDEXFROMRIGHT(2*stride*LID))]);

		// update values vor next iteration
		stride /= 2;
	} while(stride > 0);


	//////////////////////////////////////////////////////
	// round-up: make inclusive and store	//////////////
	//////////////////////////////////////////////////////
	barrier(CLK_LOCAL_MEM_FENCE);
	localBlock[OFFSET(LID)] += array[subArrayStart + LID];
	localBlock[OFFSET(local_size + LID)] += array[subArrayStart + local_size + LID];
	
	array[subArrayStart + LID] = localBlock[OFFSET(LID)];
	array[subArrayStart + local_size + LID] = localBlock[OFFSET(local_size + LID)];

	// the last result of each group is to be written in the next higher level
	// last workgroup item is responsible for that
	if (LID == local_size - 1) {
		higherLevelArray[get_group_id(0)] = localBlock[OFFSET(local_size + LID)];
		// printf("%i.", localBlock[OFFSET(local_size + LID)]); 		// debug
	}

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficientAdd(__global uint* higherLevelArray, __global uint* array, __local uint* localBlock) 
{
	// Kernel that should add the group PPS to the local PPS (Figure 14)
	int LID = get_local_id(0);
	int GrID = get_group_id(0);
	int local_size = get_local_size(0);

	if (GrID == 0) {
		return;			// no add needed
	} else {
		uint group_pps = higherLevelArray[GrID - 1];
		int subArrayStart = GrID * 2 * local_size;
		// if (LID == 0) printf("%i.", group_pps);		// debug
		array[subArrayStart + LID] += group_pps;
		array[subArrayStart + local_size + LID] += group_pps;
	}
}