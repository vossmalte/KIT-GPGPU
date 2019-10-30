
// Rotate the matrix CLOCKWISE

//naive implementation: move the elements of the matrix directly to their destinations
//this will cause unaligned memory accessed which - as we will see - should be avoided on the GPU

__kernel void MatrixRotNaive(__global const float* M, __global float* MR, uint SizeX, uint SizeY)
{
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	int x = GID.x;
	int y = GID.y;
	// printf("%i,%i\n", x,y);		// debug purpose
	MR[(x * SizeY) + (SizeY - y - 1) ] = M[x + y * SizeX];
}

//this kernel does the same thing, however, the local memory is used to
//transform a small chunk of the matrix locally
//then write it back after synchronization in a coalesced access pattern

__kernel void MatrixRotOptimized(__global const float* M, __global float* MR, uint SizeX, uint SizeY,
							__local float* block)
{
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);
	int2 block_size;
	block_size.x = get_local_size(0);
	block_size.y = get_local_size(1);
	int2 GROUPID;
	// Tile from which to read
	GROUPID.x = get_group_id(0);
	GROUPID.y = get_group_id(1);
	
	int2 TILE;
	// Tile to which to write to
	TILE.x = get_num_groups(1) - 1 - GROUPID.y;
	TILE.y = GROUPID.x;
	// if (LID.x == 0 && LID.y==0){ printf("%i,%i\n", GROUPID.x, GROUPID.y); }

	// printf("%i,%i\n", LID.x, LID.y);		// debug purpose

	// this is nice reading because of horizontally aligned cells
	block[LID.y * get_local_size(0) + LID.x] = M[GID.y * SizeX + GID.x];
	// now the unrotated tile is in local memory

	// wait for other local threads writing this array
	barrier(CLK_LOCAL_MEM_FENCE);

	// upper left corner of the rotated tile
	int2 global_rotated_pivot;
	global_rotated_pivot.x = TILE.x * block_size.y;
	global_rotated_pivot.y = TILE.y * block_size.x;
	int pivot = global_rotated_pivot.y * SizeY + global_rotated_pivot.x;

	// if (LID.x == 0) { printf("%i,%i\n", global_rotated_pivot.x, global_rotated_pivot.y); printf("%i\n", LID.y); }


	int x = LID.x;
	int y = LID.y;

	int a = 1;
	int b = 1;
	// different x / ydimensions. Due to rotation some index magic has to happen
	if (block_size.x > block_size.y) {
		x = x % block_size.y;				// truncate x to fit in a new line
		a = LID.x / block_size.y;			// x is in line a of the rotated matrix
		b = block_size.x / block_size.y;	// break down this x-line into b rotated lines
		y = b * y + a;
	}

	int rotatedX = y;
	int rotatedY = block_size.y - 1 - x;
	MR[pivot + y * SizeY + x] = block[rotatedY * block_size.x + rotatedX];


	//MR[GID.y]
}
 