
#define VALUE_TYPE float


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


__kernel void init_offsets(uint N, __global uint* in_offsets)
{
	for (int id =  get_global_id(0); id < N; id += get_global_size(0) ) 
		in_offsets[id] = id;
}

__kernel void assign_buckets(__global VALUE_TYPE* in, __global uint* in_offsets, uint N, // real scan size
							uint loopN, 
							VALUE_TYPE base_value, 
							VALUE_TYPE pivot,
							uint  num_buckets, 
							__global uint* assigned_buckets) 
{
	for (uint id = get_global_id(0); id < loopN; id+= get_global_size(0) )
	{
		int new_bucket = select( num_buckets,(uint) ((in[in_offsets[id]]  - base_value)/pivot), id < N);
		assigned_buckets[id] = select(33, new_bucket, id < N);
	}
}
 

__kernel void scan_buckets(__global VALUE_TYPE* in,  
						  __global uint* in_offsets,
						  uint N, // real scan size
						  uint loopN, 
						  __local uint* reduce_buffer, // workgroup size
						  VALUE_TYPE base_value,
						  VALUE_TYPE pivot, 
						  uint num_buckets,
						  __global uint* global_histogram) 
{
	int lid = get_local_id(0);
	if (lid == 0)
	{
		for (uint cur_bucket = 0; cur_bucket < num_buckets; ++cur_bucket)
			global_histogram[cur_bucket * get_num_groups(0) + get_group_id(0)]=0;
	}


	for (uint id = get_global_id(0); id < loopN; id+= get_global_size(0) )
	{
		uint new_bucket = select( num_buckets,(uint) ((in[in_offsets[id]]  - base_value) / pivot) , id < N);
		for (uint cur_bucket = 0; cur_bucket < num_buckets; ++cur_bucket)
		{
			reduce_buffer[lid] = select(0, 1, new_bucket  == cur_bucket);
			barrier(CLK_LOCAL_MEM_FENCE);
			workgroup_reduce(reduce_buffer);
			if (lid ==0)
				global_histogram[cur_bucket * get_num_groups(0) + get_group_id(0)] += reduce_buffer[0];
		}
	}
}

__kernel void scatter_buckets(__global VALUE_TYPE* in,  
						  __global uint* in_offsets,
						  uint N, // real scan size
						  uint loopN, 
						  __local uint* reduce_buffer, // workgroup size
						  VALUE_TYPE base_value,
						  VALUE_TYPE pivot, 
						  uint num_buckets,
						  uint end_bucket,
						  __global uint* global_histogram,
						  __global uint* out_offsets )
{
	int lid = get_local_id(0);
	for (uint id = get_global_id(0); id < loopN; id+= get_global_size(0) )
	{
		uint new_bucket = select(num_buckets, (uint)((in[in_offsets[id]]  - base_value) / pivot), id < N);
		for (uint cur_bucket = 0; cur_bucket <= end_bucket; ++cur_bucket)
		{
			reduce_buffer[lid] = select(0, 1, new_bucket  == cur_bucket);
			barrier(CLK_LOCAL_MEM_FENCE);
			workgroup_scan(reduce_buffer);
			barrier(CLK_LOCAL_MEM_FENCE);
			if (id < N && new_bucket == cur_bucket)
			{
				uint offset = reduce_buffer[lid] +  global_histogram[ cur_bucket * get_num_groups(0) + get_group_id(0) ] - 1;
				out_offsets[id] = in_offsets[offset];
			}
			if (lid ==0)
				global_histogram[cur_bucket * get_num_groups(0) + get_group_id(0)] += (reduce_buffer[get_local_size(0) -1] -1);
			
		}
	}
}
