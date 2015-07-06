#pragma OPENCL EXTENSION cl_amd_printf : enable

#define MASK 0xF
#define MASK_SIZE 4
#define VALUE_TYPE uint


inline void workgroup_reduce(uint* data) 
{
	int lid = get_local_id(0);
	int size = get_local_size(0);
	for (int d = size >>1 ;d > 0 ; d >>=1)
	{
		barrier(CLK_LOCAL_MEM_FENCE); 
	 	if (lid < d) 
	 	{
	 		data[lid] += data[lid + d];
	 	}
	 }
	 barrier(CLK_LOCAL_MEM_FENCE); 
}

// replacement for opencl 2.0 workgroup function
// expects local wg loaded into block
inline void workgroup_scan(uint* block) 
{
	int localId = get_local_id(0);
	int length = get_local_size(0);
	
	int offset = 1;
	for(int l = length>>1; l > 0; l >>= 1)
	{
	  barrier(CLK_LOCAL_MEM_FENCE);
	  if(localId < l) 
	  {
            int ai = offset*(2*localId + 1) - 1;
            int bi = offset*(2*localId + 2) - 1;
            block[bi] += block[ai];
       }
       offset <<= 1;
	}

	if(offset < length) { offset <<= 1; }

	// Build down tree
	int maxThread = offset>>1;
	for(int d = 0; d < maxThread; d<<=1)
    {
		d += 1;
		offset >>=1;
		barrier(CLK_LOCAL_MEM_FENCE);

		if(localId < d) {
			int ai = offset*(localId + 1) - 1;
			int bi = ai + (offset>>1);
		    block[bi] += block[ai];
		}
    }
	barrier(CLK_LOCAL_MEM_FENCE);
}





inline uint extract_char(VALUE_TYPE val, uchar mask, uint shift) 
{
	//long test = as_long(val);
	//long res = test ^ (-(test >> 63) | 0x8000000000000000 );
	return (val >> shift) & mask;
}

inline void create_scans(uint digit, 	
							uint start,  
							uint N,  
							uint* reduce_buffer, 
							uint shift, 
							uint first_digit,
							uint last_digit
							)
{
	int lid = get_local_id(0);
	for (uint cur_digit = first_digit; cur_digit <= last_digit; ++cur_digit)
	{
		reduce_buffer[cur_digit * get_local_size(0) + lid] = select(0,1, digit==cur_digit); // reset scans
		barrier(CLK_LOCAL_MEM_FENCE);
		workgroup_scan(&reduce_buffer[cur_digit * get_local_size(0)]);
	}

}

__kernel void init_offsets(uint N, __global uint* in_offsets)
{
	for (int id =  get_global_id(0); id < N; id += get_global_size(0) ) 
		in_offsets[id] = id;
}

__kernel void scan_digits(__global VALUE_TYPE* in,  
						  __global uint* in_offsets,
						  uint start, 
						  uint N, // real scan size
						  __local uint* reduce_buffer, 
						  uint shift, 
						  __global uint* global_histogram) 
{
	int lid = get_local_id(0);
	uint digit_counts[MASK+1] = {};// local buffer for the scanned characters
	for (int id = start + get_global_id(0); id < N; id += get_global_size(0) ) 
	{
		uint digit = select( (uint)0xF, extract_char(in[in_offsets[id]], MASK , MASK_SIZE * shift), id < N);
		digit_counts[ digit]+=1;
	}
	
	for (int digit = 0; digit < MASK+1 ; ++digit ) 
	{
		reduce_buffer[lid] = digit_counts[digit];
		barrier(CLK_LOCAL_MEM_FENCE);
		workgroup_reduce(reduce_buffer);
		if (lid == 0)	
			global_histogram[ get_group_id(0) + digit * get_num_groups(0)] = reduce_buffer[0];
	}
}


__kernel void scatter_digits(__global VALUE_TYPE*  in, 
							 __global uint* in_offsets,
								uint start, 
								uint N, 
								uint loopN,
								__local uint* reduce_buffer,
								uint shift,
								uint first_digit, 
								uint last_digit,
								__global uint* global_histogram,
								__global uint* out_offsets)
{
	for (int id = start + get_global_id(0); id < loopN; id += get_global_size(0) )
	{
		uint offset = in_offsets[id];
		uint digit = select((uint)0xF, extract_char( in[offset], MASK , MASK_SIZE * shift), id < N);
		create_scans(digit, start, N, reduce_buffer, shift, first_digit, last_digit);
		if (id < N)
		{
			uint scatter_pos  = global_histogram[digit * get_num_groups(0) + get_group_id(0)] + reduce_buffer[digit*get_local_size(0) + get_local_id(0)] -1;
			out_offsets[scatter_pos] = offset;
		}
		
		if (get_local_id(0) == 0) // pad global histogram with reduce results
		{
			for (uint cur_digit = first_digit; cur_digit <= last_digit; ++cur_digit)
			{
				global_histogram[cur_digit * get_num_groups(0) + get_group_id(0)] += (reduce_buffer[(cur_digit+1)* get_local_size(0) -1] -1);
			}
		}
	}
	
}

 