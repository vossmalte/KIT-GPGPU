
//Each thread load exactly one halo pixel
//Thus, we assume that the halo size is not larger than the 
//dimension of the work-group in the direction of the kernel

//to efficiently reduce the memory transfer overhead of the global memory
// (each pixel is lodaded multiple times at high overlaps)
// one work-item will compute RESULT_STEPS pixels

//for unrolling loops, these values have to be known at compile time

/* These macros will be defined dynamically during building the program

#define KERNEL_RADIUS 2

//horizontal kernel
#define H_GROUPSIZE_X		32
#define H_GROUPSIZE_Y		4
#define H_RESULT_STEPS		2

//vertical kernel
#define V_GROUPSIZE_X		32
#define V_GROUPSIZE_Y		16
#define V_RESULT_STEPS		3

*/

#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)


//////////////////////////////////////////////////////////////////////////////////////////////////////
// Horizontal convolution filter

/*
c_Kernel stores 2 * KERNEL_RADIUS + 1 weights, use these during the convolution
*/

//require matching work-group size
__kernel __attribute__((reqd_work_group_size(H_GROUPSIZE_X, H_GROUPSIZE_Y, 1)))
void ConvHorizontal(
			__global float* d_Dst,
			__global const float* d_Src,
			__constant float* c_Kernel,
			int Width,
			int Pitch
			)
{
	//The size of the local memory: one value for each work-item.
	//We even load unused pixels to the halo area, to keep the code and local memory access simple.
	//Since these loads are coalesced, they introduce no overhead, except for slightly redundant local memory allocation.
	//Each work-item loads H_RESULT_STEPS values + 2 halo values
	__local float tile[H_GROUPSIZE_Y][(H_RESULT_STEPS + 2) * H_GROUPSIZE_X];

	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);

	int GrID = get_group_id(0);

	// TODO:
	//const int baseX = ...
	//const int baseY = ...
	//const int offset = ...
	const int baseX = GrID * H_GROUPSIZE_X * H_RESULT_STEPS;
	// no need for baseY as this can be handled via GID.y

	// Load left halo (check for left bound)
	if (baseX == 0)	{		// left most group -> touches left bound
		tile[LID.y][LID.x] = 0;
	} else {
		tile[LID.y][LID.x] = d_Src[GID.y * Pitch + baseX - H_GROUPSIZE_X +LID.x];
	}
	//if (GrID == 1) printf("%f ", tile[LID.y][LID.x]);

	// Load main data + right halo (check for right bound)
	// for (int tileID = 1; tileID < ...)
	for (int tileID = 1; tileID < H_RESULT_STEPS + 2; tileID++) {
		int global_x = baseX + (tileID-1)*H_GROUPSIZE_X + LID.x;
		if (global_x < Width) {				// pixel readable
			tile[LID.y][LID.x+tileID*H_GROUPSIZE_X] = d_Src[GID.y * Pitch + global_x];
		} else {	// right most group -> touches right bound
			tile[LID.y][LID.x+tileID*H_GROUPSIZE_X] = 0;
		}
		// if (GID.x + GID.y == 3) printf("%i ", global_x);
	}

	// Sync the work-items after loading
	barrier(CLK_LOCAL_MEM_FENCE);

	// Convolve and store the result
	// if (GID.x + GID.y == 0) printf("Steps %i\n", H_RESULT_STEPS);
	for (int tileID = 1; tileID < H_RESULT_STEPS + 1; tileID++) {
		// if (GID.x + GID.y == 0) printf("Tile %i: ", tileID);
		int global_x = baseX + (tileID-1)*H_GROUPSIZE_X + LID.x;
		int local_x = tileID*H_GROUPSIZE_X + LID.x;

		// convolve:
		__private float px = 0;
		for (int i = 0; i < KERNEL_LENGTH; i++) {
			// if (GID.x + GID.y == 0) printf("%.1f ",c_Kernel[i]);
			px += c_Kernel[i] * tile[LID.y][local_x - KERNEL_RADIUS + i];
		}

		// if (GID.x + GID.y == 0) printf("result: %.1f \n", px);
		// store
		if (global_x < Width && GID.y < 575) {				// pixel readable
			//if (global_x == 779) printf("(%i,%i):%.1f\n", GID, px);
			d_Dst[(GID.y) * Pitch + global_x] = px;
			//d_Dst[(GID.y) * Pitch + global_x] = tile[LID.y][local_x];
			//d_Dst[GID.y * Pitch + global_x] = d_Src[GID.y * Pitch + global_x];		// no conv
		}
	}
	// if (GID.x + GID.y == 0) printf("Horizontal conv finished\n");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Vertical convolution filter

//require matching work-group size
__kernel __attribute__((reqd_work_group_size(V_GROUPSIZE_X, V_GROUPSIZE_Y, 1)))
void ConvVertical(
			__global float* d_Dst,
			__global const float* d_Src,
			__constant float* c_Kernel,
			int Height,
			int Pitch
			)
{
	__local float tile[(V_RESULT_STEPS + 2) * V_GROUPSIZE_Y][V_GROUPSIZE_X];

	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);

	int GrID = get_group_id(1);

	const int baseY = GrID * V_GROUPSIZE_Y * V_RESULT_STEPS;

	//TO DO:
	// Conceptually similar to ConvHorizontal
	// Load top halo + main data + bottom halo
	if (baseY == 0) {
		tile[LID.y][LID.x] = 0;
	} else {
		tile[LID.y][LID.x] = d_Src[(baseY - V_GROUPSIZE_Y + LID.y) * Pitch + GID.x];
	}

	for (int tileID = 1; tileID < V_RESULT_STEPS + 2; tileID++) {
		int global_y = baseY + (tileID-1)*V_GROUPSIZE_Y + LID.y;
		if (global_y < Height) {
			tile[LID.y + tileID*V_GROUPSIZE_Y][LID.x] = d_Src[global_y * Pitch + GID.x];
		} else {
			tile[LID.y + tileID*V_GROUPSIZE_Y][LID.x] = 0;
		}
	}

	// Sync the work-items after loading
	barrier(CLK_LOCAL_MEM_FENCE);

	// Compute and store results
	// Convolve and store the result

	// if (GID.x + GID.y == 0) printf("Steps %i\n", V_RESULT_STEPS);
	for (int tileID = 1; tileID < V_RESULT_STEPS + 1; tileID++) {
		// if (GID.x + GID.y == 0) printf("Tile %i: ", tileID);
		int global_y = baseY + (tileID-1)*V_GROUPSIZE_Y + LID.y;
		int local_y = tileID*V_GROUPSIZE_Y + LID.y;

		// convolve:
		float px = 0;
		for (int i = 0; i < KERNEL_LENGTH; i++) {
			// if (GID.x + GID.y == 0) printf("%.1f ",c_Kernel[i]);
			px += c_Kernel[i] * tile[local_y - KERNEL_RADIUS + i][LID.x];
		}
		// if (GID.x + GID.y == 0) printf("result(%i): %.1f \n", local_y,px);

		// store
		if (global_y < Height) {				// pixel readable
			//if (LID.x==0 ||LID.y == 0) px = 1;
			//if ((LID.x==0 ||LID.y == 0)&&tileID==1) px = 0;
			d_Dst[global_y * Pitch + GID.x] = px;
			//d_Dst[global_y * Pitch + GID.x] = tile[local_y][LID.x];
			//d_Dst[global_y * Pitch + GID.x] = d_Src[global_y * Pitch + GID.x];		// no conv
		}
	}
	// if (GID.x + GID.y == 0) printf("Vertical conv finished\n");
}
