#ifndef RADIX_SELECT_HPP
#define RADIX_SELECT_HPP

#include "viennacl/context.hpp"
#include "viennacl/vector.hpp"
#include <sstream>
#include <stdint.h>


namespace ns_radix_select
{
	struct generate_value_type
	{
		template <typename T>
		std::string type()
		{
			return T::unimplemented;
		}

		template <>
		std::string type<uint32_t>()
		{
			return "#define VALUE_TYPE uint\n";
		}


		template <>
		std::string type<uint64_t>()
		{
			return "#define VALUE_TYPE uint\n";
		}

		template <>
		std::string type < double >()
		{
			return "#define VALUE_TYPE double\n";
		}

		template <>
		std::string type<float>()
		{
			return "#define VALUE_TYPE float\n";
		}

	};




	struct generate_extract_char_kernel
	{
		template<typename basic_type>
		std::string type()
		{
			return "inline uint extract_char(VALUE_TYPE val, uchar mask, uint shift)\n"
				"{\n"
				//long test = as_long(val);
				//long res = test ^ (-(test >> 63) | 0x8000000000000000 );
				"		return (val >> shift) & mask;\n"
				"}\n";
		}

		template <>
		std::string type<double>()
		{
			return "inline uint extract_char(double val, uchar mask, uint shift)\n"
				"{\n"
				" long test = as_long(val);\n "
				" long res = test ^ (-(test >> 63) | 0x8000000000000000 );\n"
				"		return (res >> shift) & mask;\n"
				"}\n";
		}

		template <>
		std::string type<float>()
		{
			return "inline uint extract_char(float val, uchar mask, uint shift)\n"
				"{\n"
				" uint res = as_uint(val);\n "
				"		return (res >> shift) & mask;\n"
				"}\n";
		}


	};



	template <typename basic_type>
	std::string generate_kernel()
	{
		std::stringstream str;
		generate_value_type value_type;
		generate_extract_char_kernel extract_char;
		str << "#define MASK 0xF" << std::endl
			<< "#define MASK_SIZE 4" << std::endl
			<< value_type.type<basic_type>() << std::endl

			<< "inline void workgroup_reduce(uint* data)" << std::endl
			<< "{" << std::endl
			<< "	int lid = get_local_id(0); " << std::endl
			<< "	int size = get_local_size(0); " << std::endl
			<< "	for (int d = size >> 1; d > 0; d >>= 1)" << std::endl
			<< "	{" << std::endl
			<< "		barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
			<< "		if (lid < d)" << std::endl
			<< "		{" << std::endl
			<< "			data[lid] += data[lid + d];" << std::endl
			<< "		}" << std::endl
			<< "	}" << std::endl
			<< "	barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
			<< "}" << std::endl
			<< "inline void workgroup_scan(uint* block)" << std::endl
			<< "{" << std::endl
			<< "int localId = get_local_id(0);" << std::endl
			<< "int length = get_local_size(0);" << std::endl
			<< "" << std::endl
			<< "int offset = 1;" << std::endl
			<< "for (int l = length >> 1; l > 0; l >>= 1)" << std::endl
			<< "{" << std::endl
			<< "			barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
			<< "			if (localId < l)" << std::endl
			<< "			{" << std::endl
			<< "				int ai = offset*(2 * localId + 1) - 1;" << std::endl
			<< "				int bi = offset*(2 * localId + 2) - 1;" << std::endl
			<< "				block[bi] += block[ai];" << std::endl
			<< "			}" << std::endl
			<< "			offset <<= 1;" << std::endl
			<< "		}" << std::endl

			<< "		if (offset < length) { offset <<= 1; }" << std::endl

			<< "		// Build down tree" << std::endl
			<< "		int maxThread = offset >> 1;" << std::endl
			<< "		for (int d = 0; d < maxThread; d <<= 1)" << std::endl
			<< "		{" << std::endl
			<< "			d += 1;" << std::endl
			<< "			offset >>= 1;" << std::endl
			<< "			barrier(CLK_LOCAL_MEM_FENCE);" << std::endl

			<< "			if (localId < d) {" << std::endl
			<< "				int ai = offset*(localId + 1) - 1;" << std::endl
			<< "				int bi = ai + (offset >> 1);" << std::endl
			<< "				block[bi] += block[ai];" << std::endl
			<< "			}" << std::endl
			<< "		}" << std::endl
			<< "		barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
			<< "}" << std::endl


			<< extract_char.type<basic_type>() << std::endl



			<< "inline void create_scans(uint digit," << std::endl
			<< "uint start," << std::endl
			<< "		uint N," << std::endl
			<< "		uint* reduce_buffer," << std::endl
			<< "		uint shift," << std::endl
			<< "		uint first_digit," << std::endl
			<< "		uint last_digit" << std::endl
			<< "		)" << std::endl
			<< "	{" << std::endl
			<< "		int lid = get_local_id(0);" << std::endl
			<< "		for (uint cur_digit = first_digit; cur_digit <= last_digit; ++cur_digit)" << std::endl
			<< "		{" << std::endl
			<< "			reduce_buffer[cur_digit * get_local_size(0) + lid] = select(0, 1, digit == cur_digit); // reset scans" << std::endl
			<< "			barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
			<< "			workgroup_scan(&reduce_buffer[cur_digit * get_local_size(0)]);" << std::endl
			<< "		}" << std::endl

			<< "	}" << std::endl

			<< "	__kernel void init_offsets(uint N, __global uint* in_offsets)" << std::endl
			<< "	{" << std::endl
			<< "		for (int id = get_global_id(0); id < N; id += get_global_size(0))" << std::endl
			<< "			in_offsets[id] = id;" << std::endl
			<< "	}" << std::endl

			<< "	__kernel void scan_digits(__global VALUE_TYPE* in," << std::endl
			<< "		__global uint* in_offsets," << std::endl
			<< "		__global uint* scan_range," << std::endl
			<< "		__local uint* reduce_buffer," << std::endl
			<< "		uint shift," << std::endl
			<< "		__global uint* global_histogram)" << std::endl
			<< "	{" << std::endl
			<< "        if (scan_range[2] == 0) return;" << std::endl
			<< "		uint start = scan_range[0]; uint N = scan_range[1]; " << std::endl
			<< "		int lid = get_local_id(0);" << std::endl
			<< "		uint digit_counts[MASK + 1] = {};// local buffer for the scanned characters" << std::endl
			<< "		for (int id = start + get_global_id(0); id < N; id += get_global_size(0))" << std::endl
			<< "		{" << std::endl
			<< "			uint digit = select((uint)0xF, extract_char(in[id], MASK, MASK_SIZE * shift), id < N);" << std::endl
			<< "			digit_counts[digit] += 1;" << std::endl
			<< "		}" << std::endl

			<< "		for (int digit = 0; digit < MASK + 1; ++digit)" << std::endl
			<< "{" << std::endl
			<< "			reduce_buffer[lid] = digit_counts[digit];" << std::endl
			<< "			barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
			<< "			workgroup_reduce(reduce_buffer);" << std::endl
			<< "			if (lid == 0)" << std::endl
			<< "				global_histogram[get_group_id(0) + digit * get_num_groups(0)] = reduce_buffer[0];" << std::endl
			<< "		}" << std::endl
			<< "	}" << std::endl


			<< "	__kernel void scatter_digits(__global VALUE_TYPE*  in," << std::endl
			<< "		__global uint* in_offsets," << std::endl
			<< "		__global uint* scan_range," << std::endl
			<< "		__local uint* reduce_buffer," << std::endl
			<< "		uint shift," << std::endl
			<< "		__global uint* global_histogram," << std::endl
			<< "		__global uint* out_offsets, " << std::endl
			<< "		__global VALUE_TYPE* out )" << std::endl
			<< "	{" << std::endl
			<< "        if (scan_range[2] == 0) return;" << std::endl
			<< "		uint start = scan_range[0];" << std::endl
			<< "		uint N = scan_range[1];    " << std::endl
			<< "        uint K = scan_range[2];" << std::endl
			<< "		uint loopN = (N-start)/get_local_size(0); loopN = select(loopN+1, loopN, loopN*get_local_size(0)== start-N); loopN *= get_local_size(0);" << std::endl
			<< "        int last_digit = 1;" << std::endl
			<< "        for (; last_digit < 0xF && global_histogram[get_num_groups(0) * last_digit] < K; ++last_digit); " << std::endl
			<< "        uint hist_max = global_histogram[get_num_groups(0) * last_digit]; uint hist_min = global_histogram[get_num_groups(0) * (last_digit - 1)]; " << std::endl
			//<< "        if (hist_min == 0 && (last_digit < 0xF - 1 && hist_max == global_histogram[get_num_groups(0) * (last_digit + 1)])) return; " << std::endl
			// debug line
			//<< " scan_range[7] = global_histogram[get_num_groups(0) * last_digit];  scan_range[6] = global_histogram[get_num_groups(0) * (last_digit - 1)]; " << std::endl
			<< "        for (int id = get_global_id(0); id < start; id+= get_global_size(0))" << std::endl
			<< "		       {                                                            " << std::endl // clone original vector up to start
			<< "		          out[id] = in[id];                                            " << std::endl // clone original vector up to start
			<< "		          out_offsets[id] = in_offsets[id];                            " << std::endl // clone original vector up to start
			<< "		       }				                                            " << std::endl
			<< "		for (int id = get_global_id(0); id < loopN; id += get_global_size(0))" << std::endl // rescatter rest
			<< "		{" << std::endl
			<< "			uint digit = select((uint)0xF, extract_char(in[id+start], MASK, MASK_SIZE * shift), id+start < N);" << std::endl
			<< "			create_scans(digit, start, N, reduce_buffer, shift,0, last_digit);" << std::endl
			<< "			if (id+start < N && digit <= last_digit )" << std::endl
			<< "			{" << std::endl
			<< "				uint scatter_pos = start +  global_histogram[digit * get_num_groups(0) + get_group_id(0)] + reduce_buffer[digit*get_local_size(0) + get_local_id(0)] - 1;" << std::endl
			<< "				out[scatter_pos] = in[id+start];" << std::endl
			//<< "				out_offsets[id+start] = id+start;" << std::endl
			<< "				out_offsets[scatter_pos] =  in_offsets[id+start];" << std::endl
			<< "		}" << std::endl

			<< "			if (get_local_id(0) == 0) // pad global histogram with reduce results" << std::endl
			<< "			{" << std::endl
			<< "				for (uint cur_digit = 0; cur_digit <= last_digit; ++cur_digit)" << std::endl
			<< "				{" << std::endl
			<< "					global_histogram[cur_digit * get_num_groups(0) + get_group_id(0)] += (reduce_buffer[(cur_digit + 1)* get_local_size(0) - 1] );" << std::endl
			<< "				}" << std::endl
			<< "			}" << std::endl
			<< "	}" << std::endl

			<< " if (get_global_id(0) == 0 ) {" << std::endl
			// debug line
			//<< "  scan_range[3] = hist_min; scan_range[4] = hist_max; scan_range[5] = last_digit;" << std::endl
			<< "   if (hist_max == K) scan_range[2] = 0; " << std::endl
			<< "   if (hist_min+start < N && hist_min > scan_range[2]) scan_range[1] = hist_min+start; " << std::endl
			<< "   if (hist_max> scan_range[2] && hist_min < scan_range[2])" << std::endl
			<< "   {" << std::endl
			<< "	 if (hist_min > 0)" << std::endl
			<< "		scan_range[2] -= (hist_min - start);" << std::endl
			<< "	  scan_range[0] += hist_min;" << std::endl
			<< "	  scan_range[1] = start+ hist_max;" << std::endl
			<< "   }" << std::endl
			<< "}" << std::endl
			<< "}" << std::endl;
		return str.str();
	}


	template <typename T>
	void _print_vector(const viennacl::vector<T>&v, int size)
	{
		std::cout << std::endl << "--------------------------------" << std::endl;
		for (int i = 0; i < size; ++i)
		{
			std::cout << " " << std::hex << v(i);
		}
		std::cout << std::endl << "--------------------------------" << std::endl;
	}

	template <typename T>
	void _print_vector(const viennacl::vector<T>&v, int start, int end)
	{
		std::cout << std::endl << "--------------------------------" << std::endl;
		for (int i = start; i < end; ++i)
		{
			std::cout << " " << std::hex << v(i);
		}
		std::cout << std::endl << "--------------------------------" << std::endl;
	}

}

/*
	Select N smallest values from basic_type vector
	*/
template<typename basic_type>
struct radix_select
{
	viennacl::vector<unsigned int> src;
	viennacl::vector<unsigned int> dst;
	viennacl::vector<basic_type> tmp;
	viennacl::vector<unsigned int> global_histogram; 
	viennacl::vector<unsigned int> global_histogram_prefix; 

	viennacl::ocl::kernel scan_kernel;
	viennacl::ocl::kernel scatter_kernel;
	viennacl::ocl::kernel init_offsets_kernel;
	int num_groups;
	int wg_size;

	radix_select(int size, const viennacl::context& ctx)
	{
		using namespace ns_radix_select;
		static bool init = false;
		wg_size = 128;
		static int num_gpu_groups = 0;
		if (!init)
		{
			std::string program_text = generate_kernel<basic_type>();
			
			viennacl::ocl::context& ocl_ctx = const_cast<viennacl::ocl::context&>(ctx.opencl_context());
			std::cout << "Device " << ocl_ctx.current_device().name() << std::endl;
			ocl_ctx.build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
			ocl_ctx.add_program(program_text, std::string("radix_select"));
			num_gpu_groups = ocl_ctx.current_device().max_compute_units() * 4 + 1;
			init = true;
		}
		num_groups = std::min(num_gpu_groups, (int)(size / wg_size) + 1);

		src = viennacl::vector<unsigned int>(size,ctx);
		dst = viennacl::vector<unsigned int>(size, ctx);
		tmp = viennacl::vector<basic_type>(size, ctx);
		global_histogram = viennacl::vector<unsigned int>((0xF + 1) * num_groups,ctx);
		global_histogram_prefix = viennacl::vector<unsigned int>((0xF + 1) * num_groups + 1, ctx);


		scan_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "scan_digits");
		scatter_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "scatter_digits");
		init_offsets_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "init_offsets");


		


		scan_kernel.local_work_size(0, wg_size);
		scan_kernel.global_work_size(0, wg_size * num_groups);

		scatter_kernel.local_work_size(0, wg_size);
		scatter_kernel.global_work_size(0, wg_size * num_groups);

		init_offsets_kernel.local_work_size(0, wg_size);
		init_offsets_kernel.global_work_size(0, wg_size* num_groups);

	}

	viennacl::vector<unsigned int> select(int N, viennacl::vector<basic_type>& in)
	{
		int selectN = N;
		
		
		static int wg_size = 128;
		static int num_digits = 16;
		
		cl_uint size = src.size();
		viennacl::ocl::enqueue(init_offsets_kernel(size, src));
		viennacl::vector<unsigned int> scan_range(4, viennacl::traits::context(in));
		static  std::vector<unsigned int> scan_range_cpu(4);
		// test
		scan_range_cpu[0] = 0;
		scan_range_cpu[1] = in.size();
		scan_range_cpu[2] = N;
		scan_range_cpu[3] = 0;
		viennacl::copy(scan_range_cpu, scan_range);
		int shift = sizeof(basic_type) * 2 - 1;
		for (; shift >= 0; --shift)
		{
			viennacl::ocl::enqueue(scan_kernel(
				in,
				src,
				scan_range,
				viennacl::ocl::local_mem(sizeof(cl_uint) *wg_size),
				shift,
				global_histogram));

			viennacl::linalg::exclusive_scan(global_histogram, global_histogram_prefix);
			viennacl::ocl::enqueue(scatter_kernel(
				in,
				src,
				scan_range,
				viennacl::ocl::local_mem(sizeof(cl_uint) *wg_size * (0xF + 1)),
				shift,
				global_histogram_prefix,
				dst,
				tmp
				));

			/*			std::cout << "Result vector before:"   << std::endl;
			_print_vector(in, scan_range(1));
			std::cout << "Result vector after:" << std::endl;
			_print_vector(tmp, scan_range(1));
			std::cout << "Scatter : " << std::endl;
			_print_vector(dst, scan_range(1));
			std::cout << "Limits After : " << std::endl;
			_print_vector(scan_range, scan_range.size());
			*/
			tmp.fast_swap(in);
			dst.fast_swap(src);
		//	if (scan_range(2) == 0)
		//		break;
		}
		//src.resize(selectN, true);
		return src;
	}
};

#endif