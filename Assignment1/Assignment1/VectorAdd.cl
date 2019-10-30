// kernel code function
__kernel void VecAdd(__global const int* a, __global const int* b, __global int* c, int numElements) {

	int GID = get_global_id(0);
	if (GID >= numElements) { return; }
	c[GID] = a[GID] + b[numElements - GID - 1];
	return;
}



