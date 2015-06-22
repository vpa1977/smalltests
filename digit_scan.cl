/* produce digit distribution */
/* parameters -  values - global list of values */
/*               N - values length */
/*               bit - position to test */
/*               output_offsets - striped offsets 
                 reduction_buffer - workgroup sized local memory buffer for offset reduction 
*/              
__kernel void digit_scan(__global double* values, 
							uint N, // size to scan
							int bit, // shift in bits
						 __global uint* output_offsets, 
						 __local uint* reduction_buffer)
{
#define MASK_16 3
#define NUM_DIGITS 16
	int lid = get_local_id(0);
	long mask = MASK_16 << bit;
	long work_item_digits[NUM_DIGITS] = {};
	for (int id = get_global_id(0); id < N; id+= get_global_size(0) ) 
	{
		ulong test = as_ulong( values[id] );
		long result = test ^ (-(test >> 63) | 0x8000000000000000 );
		int digit = (result & mask) >> bit;
		++work_item_digits[digit];
	}
	
	for (int i = 0 ;i < NUM_DIGITS ; ++i ) 
	{
		 reduction_buffer[lid] = work_item_digits[i];
		 barrier(CLK_LOCAL_MEM_FENCE);
	}
	
}