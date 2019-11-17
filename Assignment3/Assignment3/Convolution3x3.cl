/*
We assume a 3x3 (radius: 1) convolution kernel, which is not separable.
Each work-group will process a (TILE_X x TILE_Y) tile of the image.
For coalescing, TILE_X should be multiple of 16.

Instead of examining the image border for each kernel, we recommend to pad the image
to be the multiple of the given tile-size.
*/

//should be multiple of 32 on Fermi and 16 on pre-Fermi...
#define TILE_X 32 

#define TILE_Y 16

// d_Dst is the convolution of d_Src with the kernel c_Kernel
// c_Kernel is assumed to be a float[11] array of the 3x3 convolution constants, one multiplier (for normalization) and an offset (in this order!)
// With & Height are the image dimensions (should be multiple of the tile size)
__kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1)))
void Convolution(
				__global float* d_Dst,
				__global const float* d_Src,
				__constant float* c_Kernel,
				uint Width,  // Use width to check for image bounds
				uint Height,
				uint Pitch   // Use pitch for offsetting between lines
				)
{
	// OpenCL allows to allocate the local memory from 'inside' the kernel (without using the clSetKernelArg() call)
	// in a similar way to standard C.
	// the size of the local memory necessary for the convolution is the tile size + the halo area
	__local float tile[TILE_Y + 2][TILE_X + 2];

	// TO DO...
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);

	// Fill the halo with zeros
	// Work is distributed like this:
	/*
	|########################
	|						|
	|						|
	########################|
	*/
	if (LID.y == 0){					// first row
		tile[0][LID.x + 1] = 0;
		if (LID.x == TILE_X - 1) {		// upper right element
			tile[0][TILE_X + 1] = 0;
		}
	}
	if (LID.y == TILE_Y - 1) {			// last row
		tile[TILE_Y + 1][LID.x] = 0;
		if (LID.x == 0) {				// one left of bottom right element
			tile[TILE_Y + 1][TILE_X] = 0;
		}
	}
	if (LID.x == TILE_X - 1) {			// last column halo
		tile[LID.y + 1][TILE_X + 1] = 0;
		if (LID.y == TILE_Y - 1) {		// bottom right element
			tile[TILE_Y + 1][TILE_X + 1] = 0;
		}
	}
	if (LID.x == 0) {					// first column halo
		tile[LID.y][0] = 0;				
		if (LID.y == 0) {				// one above bottom left element
			tile[TILE_Y][0] = 0;
		}
	}

	// Load main filtered area from d_Src
	tile[1 + LID.y][1 + LID.x] = d_Src[GID.y * Pitch + GID.x];

	// Load halo regions from d_Src (edges and corners separately), check for image bounds!
	// like above but with special case x in the corners -> check image bounds
	/*
	x#######################x
	|						|
	|						|
	x#######################x
	*/
	
	bool readLeftHalo = GID.x > 0;
	bool readRightHalo = GID.x < Width;
	bool readUpperHalo = GID.y > 0;
	bool readLowerHalo = GID.y < Height - 1;

	if (readUpperHalo && LID.y == 0){		// first row
		tile[0][LID.x + 1] = d_Src[(GID.y-1) * Pitch + GID.x];
	}
	if (readLowerHalo && LID.y == TILE_Y - 1) {			// last row
		tile[TILE_Y + 1][LID.x + 1] = d_Src[(GID.y+1) * Pitch + GID.x];
	}
	if (readRightHalo && LID.x == TILE_X - 1) {			// last column halo
		tile[LID.y + 1][TILE_X + 1] = d_Src[(GID.y) * Pitch + GID.x + 1];
	}
	if (readLeftHalo && LID.x == 0) {					// first column halo
		tile[LID.y + 1][0] = d_Src[(GID.y) * Pitch + GID.x - 1];
	}

	// write corners
	if (LID.y == 0 && LID.y == 0) {		// (0,0) writes all corners. no optimization for only 4 writes
		if (readLeftHalo) {
			if (readUpperHalo)			// => upper left
				tile[0][0] = d_Src[(GID.y-1) * Pitch + GID.x - 1];
			if (readLowerHalo)			// => lower left
				tile[TILE_Y + 1][0] = d_Src[(GID.y + TILE_Y+1) * Pitch + GID.x - 1];
		}
		if (readRightHalo) {
			if (readUpperHalo)			// => upper right
				tile[0][TILE_X + 1] = d_Src[(GID.y-1) * Pitch + GID.x + TILE_X + 1];
			if (readLowerHalo)			// => lower right
				tile[TILE_Y + 1][TILE_X + 1] = d_Src[(GID.y + TILE_Y+1) * Pitch + GID.x + TILE_X + 1];
		}
	}

	// Sync threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform the convolution and store the convolved signal to d_Dst.
	// store convolved signal locally
	__local float convolved[TILE_Y][TILE_X];
	convolved[LID.y][LID.x] = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			convolved[LID.y][LID.x] += tile[LID.y+i][LID.x+j] * c_Kernel[3*i + j];
		}
	}
	// last parameters of the filter:
	// normalization
	convolved[LID.y][LID.x] *= c_Kernel[9];
	// offset
	convolved[LID.y][LID.x] += c_Kernel[10];

	// store
	d_Dst[GID.y * Pitch + GID.x] = convolved[LID.y][LID.x];
}