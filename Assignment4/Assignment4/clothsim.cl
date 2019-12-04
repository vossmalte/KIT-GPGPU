#define DAMPING 0.02f

#define G_ACCEL (float4)(0.f, -9.81f, 0.f, 0.f)
#define ZERO 	(float4)(0.f, 0.f, 0.f, 0.f)

#define WEIGHT_ORTHO	0.138f
#define WEIGHT_DIAG		0.097f
#define WEIGHT_ORTHO_2	0.069f
#define WEIGHT_DIAG_2	0.048f


#define ROOT_OF_2 1.4142135f
#define DOUBLE_ROOT_OF_2 2.8284271f




///////////////////////////////////////////////////////////////////////////////
// The integration kernel
// Input data:
// width and height - the dimensions of the particle grid
// d_pos - the most recent position of the cloth particle while...
// d_prevPos - ...contains the position from the previous iteration.
// elapsedTime      - contains the elapsed time since the previous invocation of the kernel,
// prevElapsedTime  - contains the previous time step.
// simulationTime   - contains the time elapsed since the start of the simulation (useful for wind)
// All time values are given in seconds.
//
// Output data:
// d_prevPos - Input data from d_pos must be copied to this array
// d_pos     - Updated positions
///////////////////////////////////////////////////////////////////////////////
  __kernel void Integrate(unsigned int width,
						unsigned int height, 
						__global float4* d_pos,
						__global float4* d_prevPos,
						float elapsedTime,
						float prevElapsedTime,
						float simulationTime) {
							
	// Make sure the work-item does not map outside the cloth
    if(get_global_id(0) >= width || get_global_id(1) >= height)
	{
		printf("(%i, %i) is not executing anything\n", get_global_id(0), get_global_id(1));
		return;
	}

	unsigned int particleID = get_global_id(0) + get_global_id(1) * width;
	// This is just to keep every 8th particle of the first row attached to the bar
    if(particleID > width-1 || ( particleID & ( 7 )) != 0){
		if (simulationTime < 0.01) {
			// initiate the prevPos
			//printf("initiating prevPos\n");
			d_prevPos[particleID] = d_pos[particleID];
		}
		if (get_global_id(0) == 2 && get_global_id(1) == 3) {
			//printf("simulationtime: %.2f, elapsed time: %.2f\n", simulationTime, elapsedTime);
			printf("%.2f, %.2f, %.2f, %.2f\n", d_prevPos[particleID]);
		}
		if (particleID == width)
		//printf("%.2f, %.2f, %.2f, %.2f\n", d_prevPos[particleID]);
		;


		// ADD YOUR CODE HERE!

		// Read the positions
		// Compute the new one position using the Verlet position integration, taking into account gravity and wind
		// velocity integration: p1 = p0 + v0*dt + 0.5a0*dt^2
		// position integration:
		// https://www.lonesock.net/article/verlet.html
		// xi+1 = xi + (xi - xi-1) * (dti / dti-1) + a * dti * dti
		float4 pos = d_pos[particleID] + (d_pos[particleID] - d_prevPos[particleID])*elapsedTime + G_ACCEL*elapsedTime*elapsedTime;
		// Move the value from d_pos into d_prevPos and store the new one in d_pos
		d_prevPos[particleID] = d_pos[particleID];
		d_pos[particleID] = pos;


    } else {
		// this is a particle at the bar
		d_prevPos[particleID] = d_pos[particleID];
	}
}



///////////////////////////////////////////////////////////////////////////////
// Input data:
// pos1 and pos2 - The positions of two particles
// restDistance  - the distance between the given particles at rest
//
// Return data:
// correction vector for particle 1
///////////////////////////////////////////////////////////////////////////////
  float4 SatisfyConstraint(float4 pos1,
						 float4 pos2,
						 float restDistance){
	if (dot(pos2,pos2) == 0.f)	// dont do anything if all zero
		return ZERO;
	float4 toNeighbor = pos2 - pos1;
	return (toNeighbor - normalize(toNeighbor) * restDistance);
}

///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// restDistance     - the distance between two orthogonally neighboring particles at rest
// d_posIn          - the input positions
//
// Output data:
// d_posOut - new positions must be written here
///////////////////////////////////////////////////////////////////////////////

#define TILE_X 16 
#define TILE_Y 16
#define HALOSIZE 2

__kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1)))
__kernel void SatisfyConstraints(unsigned int width,
								unsigned int height, 
								float restDistance,
								__global float4* d_posOut,
								__global float4 const * d_posIn){
    
    // Make sure the work-item does not map outside the cloth
    if(get_global_id(0) >= width || get_global_id(1) >= height)
	{
		printf("(%i, %i) is not executing anything\n", get_global_id(0), get_global_id(1));
		return;
	}
		

	//int TILE_X = get_local_size(0);
	//int TILE_Y = get_local_size(1);

	// use local memory as cache:
	// halo of width 2
	__local float4 tile[TILE_Y+4][TILE_X+4];

	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);

	int2 GrID;
	GrID.x = get_group_id(0);
	GrID.y = get_group_id(1);

	unsigned int particleID = get_global_id(0) + get_global_id(1) * width;

	// Fill the halo with zeros
	tile[LID.y][LID.x] = ZERO;
	tile[LID.y+4][LID.x+4] = ZERO;
	tile[LID.y+4][LID.x] = ZERO;
	tile[LID.y][LID.x+4] = ZERO;

	barrier(CLK_LOCAL_MEM_FENCE);
	// Load main filtered area from d_posIn
	tile[2 + LID.y][2 + LID.x] = d_posIn[particleID];

	
	// write halo
	bool readLeftHalo = GrID.x > 0;
	bool readRightHalo = GrID.x * TILE_X < get_num_groups(0) - 1;
	bool readUpperHalo = GrID.y > 0;
	bool readLowerHalo = GrID.y < get_num_groups(1) - 1;

	if (readUpperHalo && LID.y <= 1){		// first row
		tile[0][LID.x + 1] = d_posIn[particleID - (2-LID.y)*width];
	}
	if (readLowerHalo && LID.y >= TILE_Y - 2) {			// last row
		tile[TILE_Y + 2][LID.x + 2] = d_posIn[particleID + (LID.y - TILE_Y + 3)*width];
	}
	if (readRightHalo && LID.x >= TILE_X - 2) {			// last column halo
		tile[LID.y + 2][TILE_X + 2] = d_posIn[particleID + (LID.x - TILE_X + 3)];
	}
	if (readLeftHalo && LID.x <= 1) {					// first column halo
		tile[LID.y + 2][0] = d_posIn[particleID - 2 + LID.x];
	}

	// write corners
	if (LID.y <= 1 && LID.y <= 1) {		// upper left 2x2 writes all corners. no optimization for only 4 writes
		if (readLeftHalo) {
			if (readUpperHalo)			// => upper left
				tile[LID.y][LID.x] = d_posIn[particleID - 2 + LID.x - (2-LID.y)*width];
			if (readLowerHalo)			// => lower left
				tile[TILE_Y + 2 + LID.y][LID.x] = d_posIn[particleID - 2 + LID.x + (TILE_Y + LID.y) *width];
		}
		if (readRightHalo) {
			if (readUpperHalo)			// => upper right
				tile[LID.y][TILE_X + 2 + LID.x] = d_posIn[particleID - (2-LID.y)* width + 1 + LID.x];
			if (readLowerHalo)			// => lower right
				tile[TILE_Y + 2 + LID.y][TILE_X + 2 + LID.x] = d_posIn[particleID - (TILE_Y + LID.y) *width + 1 + LID.x];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_global_id(0) == 2 && get_global_id(1) == 3) printf("Loaded tile into local memory\n");


	// ADD YOUR CODE HERE!
	// Satisfy all the constraints (structural, shear, and bend).
	// You can use weights defined at the beginning of this file.

	float4 thisPos = tile[LID.y + 2][LID.x + 2];
	float4 cumulatedCorrection = ZERO;
	// This is just to keep every 8th particle of the first row attached to the bar
    if(particleID > width-1 || ( particleID & ( 7 )) != 0){

		// structural constraints
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2 - 1][LID.x + 2], restDistance) * WEIGHT_ORTHO;
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2][LID.x + 2 - 1], restDistance) * WEIGHT_ORTHO;
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2][LID.x + 2 + 1], restDistance) * WEIGHT_ORTHO;
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2 + 1][LID.x + 2], restDistance) * WEIGHT_ORTHO;
		if (get_global_id(0) == 2 && get_global_id(1) == 3) printf("structural\n");

		// shear constraints
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2 - 1][LID.x + 2 - 1], restDistance*ROOT_OF_2) * WEIGHT_DIAG;
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2 - 1][LID.x + 2 + 1], restDistance*ROOT_OF_2) * WEIGHT_DIAG;
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2 + 1][LID.x + 2 - 1], restDistance*ROOT_OF_2) * WEIGHT_DIAG;
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2 + 1][LID.x + 2 + 1], restDistance*ROOT_OF_2) * WEIGHT_DIAG;
		if (get_global_id(0) == 2 && get_global_id(1) == 3) printf("shear\n");

		// bend constraints
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2][LID.x + 2 - 2], restDistance*2.f) * WEIGHT_ORTHO_2;
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2][LID.x + 2 + 2], restDistance*2.f) * WEIGHT_ORTHO_2;
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2 + 2][LID.x + 2], restDistance*2.f) * WEIGHT_ORTHO_2;
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2 - 2][LID.x + 2], restDistance*2.f) * WEIGHT_ORTHO_2;
		if (get_global_id(0) == 2 && get_global_id(1) == 3) printf("bend\n");

		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2 - 2][LID.x + 2 + 2], restDistance*2.f*ROOT_OF_2) * WEIGHT_DIAG_2;
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2 - 2][LID.x + 2 - 2], restDistance*2.f*ROOT_OF_2) * WEIGHT_DIAG_2;
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2 + 2][LID.x + 2 + 2], restDistance*2.f*ROOT_OF_2) * WEIGHT_DIAG_2;
		cumulatedCorrection += SatisfyConstraint(thisPos, tile[LID.y + 2 + 2][LID.x + 2 - 2], restDistance*2.f*ROOT_OF_2) * WEIGHT_DIAG_2;

		
	}

	if (get_global_id(0) == 2 && get_global_id(1) == 3) {
			//printf("simulationtime: %.2f, elapsed time: %.2f\n", simulationTime, elapsedTime);
			printf("%.2f, %.2f, %.2f, %.2f\n", cumulatedCorrection);
		}

	d_posOut[particleID] = thisPos + cumulatedCorrection;

	// A ping-pong scheme is needed here, so read the values from d_posIn and store the results in d_posOut

	// Hint: you should use the SatisfyConstraint helper function in the following manner:
	//SatisfyConstraint(pos, neighborpos, restDistance) * WEIGHT_XXX

}


///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// d_pos            - the input positions
// spherePos        - The position of the sphere (xyz)
// sphereRad        - The radius of the sphere
//
// Output data:
// d_pos            - The updated positions
///////////////////////////////////////////////////////////////////////////////
__kernel void CheckCollisions(unsigned int width,
								unsigned int height, 
								__global float4* d_pos,
								float4 spherePos,
								float sphereRad){
								

	// ADD YOUR CODE HERE!
	// Find whether the particle is inside the sphere.
	// If so, push it outside.
	// Make sure the work-item does not map outside the cloth
    if(get_global_id(0) >= width || get_global_id(1) >= height)
		return;

	unsigned int particleID = get_global_id(0) + get_global_id(1) * width;
	// This is just to keep every 8th particle of the first row attached to the bar
    if(particleID > width-1 || ( particleID & ( 7 )) != 0){
		if (dot(d_pos[particleID] - spherePos, d_pos[particleID] - spherePos) < sphereRad * sphereRad) {
			// collision
			d_pos[particleID] = spherePos + normalize(d_pos[particleID] - spherePos) * sphereRad;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this function!
///////////////////////////////////////////////////////////////////////////////
float4 CalcTriangleNormal( float4 p1, float4 p2, float4 p3) {
    float4 v1 = p2-p1;
    float4 v2 = p3-p1;

    return cross( v1, v2);
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this kernel!
///////////////////////////////////////////////////////////////////////////////
__kernel void ComputeNormals(unsigned int width,
								unsigned int height, 
								__global float4* d_pos,
								__global float4* d_normal){
								
    int particleID = get_global_id(0) + get_global_id(1) * width;
    float4 normal = (float4)( 0.0f, 0.0f, 0.0f, 0.0f);
    
    int minX, maxX, minY, maxY, cntX, cntY;
    minX = max( (int)(0), (int)(get_global_id(0)-1));
    maxX = min( (int)(width-1), (int)(get_global_id(0)+1));
    minY = max( (int)(0), (int)(get_global_id(1)-1));
    maxY = min( (int)(height-1), (int)(get_global_id(1)+1));
    
    for( cntX = minX; cntX < maxX; ++cntX) {
        for( cntY = minY; cntY < maxY; ++cntY) {
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY+1)],
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
        }
    }
    d_normal[particleID] = normalize( normal);
}
