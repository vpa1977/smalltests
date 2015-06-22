inline void compare_and_swap(__global double2 *d1, __global double2 *d2, uint dir) {
   double2 input1 = *d1; double2 input2 = *d2;
   uint2 cmp = as_uint2(input1 > input2) ^ dir;
   uint2 mask = 2*cmp;
   mask.s1 += 1;
   *d1 = shuffle2(input1, input2, mask);
   *d2 = shuffle2(input2, input1, mask);
}

inline void compare_and_swap_scalar(__global double * local_data , uint dir) {
	if((local_data[0] > local_data[1]) == dir ) {
	   double t = local_data[0]; 
	   local_data[0] = local_data[1]; 
	   local_data[1] = t;
	}
}

__kernel void shuffle_test(__global double2*  values) 
{
	int gid = 2*get_global_id(0);
	compare_and_swap(&values[gid], &values[gid+1],1); 
}


__kernel void scalar_test(__global  double*  values) 
{
	int gid = 2*get_global_id(0);
	compare_and_swap_scalar(&values[gid],0); 
}