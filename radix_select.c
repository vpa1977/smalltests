#define MASK 0xF
#define MASK_SIZE 4
#define VALUE_TYPE int
inline uint extract_char(VALUE_TYPE val, uchar mask, uint shift)
{
//	long test = as_long(val);
	//long res = test ^ (-(test >> 63) | 0x8000000000000000);
	//return (res >> shift) & mask;
	return (val >> shift) & mask;
}


#define WG_SIZE 128
#define SCAN_WG_SIZE 128
#define SCAN_GLOBAL_SIZE 128*SCAN_WG_SIZE


struct scatter_context
{
	uint hist_min;
	uint hist_max;
	uint stop;
	uint min_digit;
	uint max_digit;
	uint shift;
	uint debug_n;
	uint debug_start;
	uint debug_end;
	uint debug_shift;
	uint debug_old_n;
	uint debug_hist_point;
};

__kernel void scan_digits(
	int N,
	__global VALUE_TYPE* in,
	__global uint* in_offsets,
	__global VALUE_TYPE* out,
	__global uint* out_offsets,
	
	uint end, // real scan size
	uint shift,
	__global uint* global_histogram_prefix,
	__global uint* carries,
	__global uint* context);

__kernel void scatter_digits(
	int N,
	__global VALUE_TYPE*  in,
	__global uint* in_offsets,
	__global VALUE_TYPE*  out,
	__global uint* out_offsets,
	uint end,
	uint shift,
	__global uint* global_histogram_prefix,
	__global uint* carries,
	__global uint* context);

__kernel void scan_with_offset
(
	__global VALUE_TYPE* in,
	__global uint* in_offsets,
	__global VALUE_TYPE* out,
	__global uint* out_offsets,
	__global uint* global_histogram_prefix,
	__global uint* carries,
	__global uint* context,
	uint parent_global_size,
	uint parent_local_size
	)
{
	struct scatter_context* ctx = (struct scatter_context*)context;
	int N = ctx->debug_n;
	uint start = ctx->debug_start;
	uint end =  ctx->debug_end;
	int shift = ctx->shift;
	VALUE_TYPE* in_param = in + start;
	uint* offset_param = in_offsets + start;
	scan_digits(N, in +start, in_offsets + start, out, out_offsets, end, shift, global_histogram_prefix, carries, context);
}




inline void workgroup_reduce(uint* data)
{
	int lid = get_local_id(0);
	int size = get_local_size(0);
	for (int d = size >> 1; d > 0; d >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < d)
		{
			data[lid] += data[lid + d];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

inline void workgroup_scan(uint* block)
{
	int localId = get_local_id(0);
	int length = get_local_size(0);

	int offset = 1;
	for (int l = length >> 1; l > 0; l >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localId < l)
		{
			int ai = offset*(2 * localId + 1) - 1;
			int bi = offset*(2 * localId + 2) - 1;
			block[bi] += block[ai];
		}
		offset <<= 1;
	}
	if (offset < length) { offset <<= 1; }
	// Build down tree
	int maxThread = offset >> 1;
	for (int d = 0; d < maxThread; d <<= 1)
	{
		d += 1;
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localId < d) {
			int ai = offset*(localId + 1) - 1;
			int bi = ai + (offset >> 1);
			block[bi] += block[ai];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}


__kernel void scan_digits_start(
	__global uint* in_offsets,
	uint end)
{
	for (int id = get_global_id(0); id < end; id += get_global_size(0))
		in_offsets[id] = id;

}

__kernel void scan_2( __global uint* carries) {
	__local uint shared_buffer[256];
	uint my_carry = carries[get_local_id(0)];
	for (unsigned int stride = 1; stride < get_local_size(0); stride *= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);
		shared_buffer[get_local_id(0)] = my_carry;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (get_local_id(0) >= stride)
			my_carry += shared_buffer[get_local_id(0) - stride];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	shared_buffer[get_local_id(0)] = my_carry;
	barrier(CLK_LOCAL_MEM_FENCE);
	carries[get_local_id(0)] = (get_local_id(0) > 0) ? shared_buffer[get_local_id(0) - 1] : 0;
}

__kernel void scan_3(__global uint * Y,
	unsigned int startY,
	unsigned int incY,
	unsigned int sizeY,
	__global uint* carries) {
	unsigned int work_per_thread = (sizeY - 1) / get_global_size(0) + 1;
	unsigned int block_start = work_per_thread * get_local_size(0) *  get_group_id(0);
	unsigned int block_stop = work_per_thread * get_local_size(0) * (get_group_id(0) + 1);
	__local uint shared_offset;
	if (get_local_id(0) == 0) shared_offset = carries[get_group_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (unsigned int i = block_start + get_local_id(0); i < block_stop; i += get_local_size(0))
		if (i < sizeY)
			Y[i * incY + startY] += shared_offset;
}


__kernel void scan_1(__global uint* X,
	unsigned int startX,
	unsigned int incX,
	unsigned int sizeX,
	__global uint* Y,
	unsigned int startY,
	unsigned int incY,
	unsigned int scan_offset,
	__global uint* carries) {
	__local uint shared_buffer[256];
	uint my_value;
	unsigned int work_per_thread = (sizeX - 1) / get_global_size(0) + 1;
	unsigned int block_start = work_per_thread * get_local_size(0) *  get_group_id(0);
	unsigned int block_stop = work_per_thread * get_local_size(0) * (get_group_id(0) + 1);
	unsigned int block_offset = 0;
	for (unsigned int i = block_start + get_local_id(0); i < block_stop; i += get_local_size(0)) {
		my_value = (i < sizeX) ? X[i * incX + startX] : 0;
		for (unsigned int stride = 1; stride < get_local_size(0); stride *= 2) {
			barrier(CLK_LOCAL_MEM_FENCE);
			shared_buffer[get_local_id(0)] = my_value;
			barrier(CLK_LOCAL_MEM_FENCE);
			if (get_local_id(0) >= stride)
				my_value += shared_buffer[get_local_id(0) - stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		shared_buffer[get_local_id(0)] = my_value;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (scan_offset > 0)
			my_value = (get_local_id(0) > 0) ? shared_buffer[get_local_id(0) - 1] : 0;
		if (i < sizeX)
			Y[i * incY + startY] = block_offset + my_value;
		block_offset += shared_buffer[get_local_size(0) - 1];
	}
	if (get_local_id(0) == 0) carries[get_group_id(0)] = block_offset;
		
}

__kernel void find_limits(
	int N,
	__global VALUE_TYPE* in,
	__global uint* in_offsets,
	__global VALUE_TYPE* out,
	__global uint* out_offsets,
	uint end, // real scan size
	uint shift,
	__global uint* global_histogram_prefix,
	__global uint* carries,
	__global uint* context, 
	uint parent_kernel_global_size,
	uint parent_kernel_local_size)
{
	struct scatter_context* ctx = (struct scatter_context*)context;
	context[get_global_id(0)] = 0;
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	uint num_groups = parent_kernel_global_size / parent_kernel_local_size;
	
	uint idx = num_groups * get_local_id(0);
	uint idx_next = select((uint)idx + 1, (uint)(num_groups* (1 + get_local_id(0))), (get_local_id(0) != 0xF));
	if (global_histogram_prefix[idx] < N && global_histogram_prefix[idx_next] >= N)
	{
		ctx->hist_min = global_histogram_prefix[idx];
		ctx->hist_max = global_histogram_prefix[idx_next];
		ctx->max_digit = get_local_id(0);
		if (global_histogram_prefix[idx] == 0 && global_histogram_prefix[idx_next] >= end)
		{
			ctx->stop = true;
		}
		else
		{
			ctx->stop = false;
		}
	}
	if (global_histogram_prefix[idx] == 0 && global_histogram_prefix[idx_next] != 0)
	{
		ctx->min_digit = get_local_id(0);
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	
	if (get_global_id(0) == 0)
	{
		if (global_histogram_prefix[num_groups * get_local_size(0)] <= N)
			ctx->stop = true;
		
		ndrange_t scan_range = ndrange_1D(parent_kernel_global_size, parent_kernel_local_size);
		if (ctx->stop || end <= N)
		{
			if (shift >0)
			{
				enqueue_kernel(get_default_queue(),
					CLK_ENQUEUE_FLAGS_NO_WAIT,
					scan_range,
					^{ scan_digits(N, in, in_offsets, out, out_offsets,  end, shift - 1,  global_histogram_prefix,carries, context); }
				); 
			}
		}
		else 
		{
			clk_event_t evt_data_scattered, evt_data_copied;
			enqueue_kernel(get_default_queue(),
				CLK_ENQUEUE_FLAGS_NO_WAIT,
				scan_range,
				0, NULL, &evt_data_scattered,
				^{ scatter_digits(N, in, in_offsets, out, out_offsets,  end, shift,   global_histogram_prefix, carries, context); }
			);

			enqueue_kernel(get_default_queue(),
				CLK_ENQUEUE_FLAGS_NO_WAIT,
				scan_range,
				1, &evt_data_scattered, &evt_data_copied,
				^{ for (int idx = get_global_id(0); idx < end; idx += get_global_size(0))
					{
						in[idx] = out[idx];
						in_offsets[idx] = out_offsets[idx];
					}
			});

			int newN = N - (ctx->hist_min);
			uint start = ctx->hist_min;
			uint new_end = min(end, ctx->hist_max);
			new_end = new_end - start;
			__global VALUE_TYPE* new_in = in + start;
			__global uint* new_offsets = in_offsets + start;
			ctx->debug_n = newN;
			ctx->debug_start = start;
			ctx->debug_end = new_end;
			ctx->shift = shift - 1;

			/*enqueue_kernel(get_default_queue(),
				CLK_ENQUEUE_FLAGS_NO_WAIT,
				scan_range,
				1, &evt_data_copied, NULL,
				^{ scan_digits(newN, new_in, new_offsets, out, out_offsets,  new_end, shift - 1,  global_histogram_prefix,carries, context); }
			);*/

			
			release_event(evt_data_scattered);
			release_event(evt_data_copied);

		}
	}
}






__kernel void scatter_digits(
	int N,
	__global VALUE_TYPE*  in,
	__global uint* in_offsets,
	__global VALUE_TYPE*  out,
	__global uint* out_offsets,
	uint end,
	uint shift,
	__global uint* global_histogram_prefix, 
	__global uint* carries,
	__global uint* context)
{
//	if (get_global_id(0) == 0 ) context[20 + shift] = 1;

	__local uint reduce_buffer[WG_SIZE][0x10];
	struct scatter_context* ctx = (struct scatter_context*)context;

	uint lid = get_local_id(0);
	uint main = (end / get_local_size(0));
	main *= get_local_size(0);
	uint loop_end = select((uint)(main + get_local_size(0)), (uint)main, main == end);

	__local uint prefix[0x10];
//	__local uint prefix_next[0x10];
	if (get_local_id(0) == 0)
	{

		for (int lid_idx = 0; lid_idx < 0x10; lid_idx += 1)
			prefix[lid_idx] = global_histogram_prefix[lid_idx * get_num_groups(0) + get_group_id(0)];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	uint loop_step = 0;
	// loop should be aligned to the workgroup size
	for (int id = get_global_id(0); id < loop_end; id += get_global_size(0), loop_step++)
	{
		// build scans for current workgroup/iteration
		uint cur_digit = select((uint)0xF, extract_char(in[id], MASK, MASK_SIZE * shift), id  < end);

		for (uint digit = 0; digit <= 0xF; ++digit)
		{
			reduce_buffer[digit][lid] = select(0, 1, digit == cur_digit); // reset scans
			
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint digit =  ctx->min_digit; digit <= ctx->max_digit; ++digit)
		{
			//if (prefix[digit] != prefix_next[digit])
			{
				workgroup_scan(&reduce_buffer[digit][0]);
			}
			//out_offsets[get_local_size(0) * 0x10 * loop_step + get_local_size(0)*digit + lid] = reduce_buffer[digit][lid];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (id  < end && cur_digit >= ctx->min_digit && cur_digit <= ctx->max_digit)
		{
			uint scatter_pos = prefix[cur_digit] + reduce_buffer[cur_digit][lid] - 1;
			out[scatter_pos ] =  in[id];
			out_offsets[scatter_pos] = in_offsets[id];
			//out_offsets[id] = scatter_pos;
		}

		if (get_local_id(0) == 0)
		{
			for (int lid_idx = 0; lid_idx < 0x10; lid_idx += 1) // pad global histogram with reduce results
				prefix[lid_idx] += reduce_buffer[lid_idx][get_local_size(0) - 1];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	
	
}



__kernel void scan_digits(
	int N,
	__global VALUE_TYPE* in,
	__global uint* in_offsets,
	__global VALUE_TYPE* out, 
	__global uint* out_offsets,
	uint end, // real scan size
	uint shift,
	__global uint* global_histogram_prefix, 
	__global uint* carries, 
	__global uint* context)
{
	//if (get_global_id(0) == 0) context[10 + shift] = 1;
	struct scatter_context* ctx = (struct scatter_context*)context;
	__local uint reduce_buffer[WG_SIZE];
	int lid = get_local_id(0);
	uint digit_counts[MASK + 1] = {};// local buffer for the scanned characters
	for (int id = get_global_id(0); id < end; id += get_global_size(0))
	{
		uint digit =  extract_char(in[id], MASK, MASK_SIZE * shift);
		digit_counts[digit] +=1;
	}
	
	for (int digit = 0; digit < 0x10; ++digit)
	{
		reduce_buffer[lid] = digit_counts[digit];
		barrier(CLK_LOCAL_MEM_FENCE);
		workgroup_reduce(reduce_buffer);
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid == 0)
			global_histogram_prefix[get_group_id(0) + digit * get_num_groups(0)] = reduce_buffer[0];
	}
	
		

	barrier(CLK_GLOBAL_MEM_FENCE);
	if (get_global_id(0) == 0)
	{
		global_histogram_prefix[0x10 * get_num_groups(0)] = reduce_buffer[get_local_size(0) - 1];
		int histogram_size = get_num_groups(0) *0x10+1;
		clk_event_t evt_histogram_ready, evt_scan_ready, evt_carry_ready, evt_ready_relaunch;
		ndrange_t sum_range = ndrange_1D(SCAN_GLOBAL_SIZE, SCAN_WG_SIZE);
		ndrange_t scatter_range = ndrange_1D(get_global_size(0), get_local_size(0));
		ndrange_t carries_range = ndrange_1D(SCAN_WG_SIZE, SCAN_WG_SIZE);
		ndrange_t limits_range = ndrange_1D(0x10, 0x10);
		uint parent_global_size = get_global_size(0);
		uint parent_local_size = get_local_size(0);
		enqueue_kernel(get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			sum_range,
			0, NULL, &evt_scan_ready,
			^{ scan_1(global_histogram_prefix, 0, 1, histogram_size, global_histogram_prefix, 0, 1, 1, carries); }
		);

		enqueue_kernel(get_default_queue(),
			CLK_ENQUEUE_FLAGS_NO_WAIT,
			carries_range,
			1, &evt_scan_ready, &evt_carry_ready,
			^{ scan_2(carries); }
		);

		enqueue_kernel(get_default_queue(),
			CLK_ENQUEUE_FLAGS_NO_WAIT,
			sum_range,
			1, &evt_carry_ready, &evt_histogram_ready,
			^{scan_3(global_histogram_prefix, 0, 1, histogram_size, carries); }
		);
		
		enqueue_kernel(get_default_queue(),
			CLK_ENQUEUE_FLAGS_NO_WAIT,
			limits_range,
			1, &evt_histogram_ready,&evt_ready_relaunch,
			^{ find_limits(N, in, in_offsets, out, out_offsets, end, shift,   global_histogram_prefix, carries, context, parent_global_size, parent_local_size); }
		);

		/*enqueue_kernel(get_default_queue(),
			CLK_ENQUEUE_FLAGS_NO_WAIT,
			ndrange_1D(1),
			1, &evt_ready_relaunch, NULL,
			^{ scan_with_offset(in, in_offsets, out, out_offsets, global_histogram_prefix, carries, context, parent_global_size, parent_local_size); }
		);*/

		release_event(evt_scan_ready);
		release_event(evt_carry_ready);
		release_event(evt_histogram_ready);
		release_event(evt_ready_relaunch);
	}

}



