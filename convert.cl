
inline long convert_to_long(double val) 
{
   long test = as_long(val);
  // return test ^ (-(test >> 63) | 0x8000000000000000 );
  return test;
}

inline uint extract_char(long val, uchar mask, uint shift) 
{
	return (val >> shift) & mask;
}

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



__kernel void workgroup_scan_test(__global uint* flags, __local uint* reduction_buffer) 
{
	int lid = get_local_id(0);
	reduction_buffer[lid] = flags[lid];
	barrier(CLK_LOCAL_MEM_FENCE);
	workgroup_scan(reduction_buffer);
	flags[lid] = reduction_buffer[lid]; 
} 

__kernel void workgroup_reduction_test(__global uint* flags, __local uint* reduction_buffer) 
{
	int lid = get_local_id(0);
	reduction_buffer[lid] = flags[lid];
	barrier(CLK_LOCAL_MEM_FENCE);
	workgroup_reduce(reduction_buffer);
	if (lid == 0 ) 
		flags[0] = reduction_buffer[0]; 
}

__kernel void convert(__global double* in, __global long* out) 
{
	out[get_global_id(0)] = convert_to_long(in[get_global_id(0)]);
}


__kernel void scatter_keys(__global uint* keys_in, uint N)
{
	for (int id = get_global_size(0); id < N; id += get_global_size(0) )
		keys_in[ get_global_id(0) ] = id;
}

#define MASK 0xF
#define MASK_SIZE 4

__kernel void scan_digits(__global double* in,  
			uint start, uint N, __local uint* reduce_buffer, uint shift, 
			__global uint* global_histogram) 
{
	int lid = get_local_id(0);
	uint digit_counts[MASK+1] = {};// local buffer for the scanned characters
	for (int id = start + get_global_id(0); id < N; id += get_global_size(0) ) 
	{
		uint digit = extract_char( convert_to_long(in[id]), MASK , MASK_SIZE * shift);
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

__kernel void scatter_digits(__local uint* local_positions,
								__global double* in, 
								__global double* out,
								__global uint* global_offset, 
							    uint start,
								uint N, 
								uint first_digit, 
								uint last_digit, 
								int shift,
 								__global uint* keys_in, 
 								__global uint* keys_out 
 								) 
{
	int lid = get_local_id(0);
	for (int id = start + get_global_id(0); id < N; id += get_global_size(0) )
	{
		uint digit = extract_char( convert_to_long(in[id]), MASK , MASK_SIZE * shift);
		for (uint cur_digit = first_digit; cur_digit <= last_digit; ++cur_digit)
		{
			local_positions[cur_digit * get_local_size(0) + lid] = select(0,1, digit==cur_digit); // reset scans
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		workgroup_scan(local_positions);
		barrier(CLK_LOCAL_MEM_FENCE);
		
		keys_out[id] = local_positions[lid];
	}
}
	/*
	for (int id = start + get_global_id(0); id < N; id += get_global_size(0) ) 
	{
		
		out[id] = digit;
		
	
		// create scans for each digit
		for (int cur_digit = first_digit; cur_digit <= last_digit; ++cur_digit)
		{
		//	local_positions[cur_digit * get_local_size(0) + lid] = select(0,1, digit==cur_digit); // reset scans 
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		
		if (digit >= first_digit && digit <= last_digit)
		{
		 //digit * get_local_size(0) +
			uint offset = local_positions[lid];// + global_offset[ get_group_id(0) + digit* get_num_groups(0)];
			//out[offset] = in[id]; // scatter value
		//	keys_out[id] = offset;
			
		}
	}
}*/
 