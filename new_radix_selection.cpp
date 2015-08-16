#include <vector>
#include "gtest/gtest.h"
#include "merge_sort.hpp"
#include "viennacl/context.hpp"
#include "viennacl/ml/knn.hpp"

#include <boost/numeric/ublas/matrix.hpp>
#include <atomic>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

#ifndef MAGIC

#define MAGIC

#include "viennacl/context.hpp"
#include "viennacl/vector.hpp"
#include <sstream>
#include <stdint.h>

namespace magic_hamster
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
	FILE * tmp = fopen("radix_select.c", "rb");
	fseek(tmp, 0, SEEK_END);
	std::vector<char> binary;
	binary.resize(ftell(tmp));
	rewind(tmp);
	fread(&binary[0], binary.size(), 1, tmp);
	fclose(tmp);
	binary.push_back(0);

	if (true)
	return std::string(&binary[0]);
	

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
		<< "		uint start," << std::endl
		<< "		uint N, // real scan size" << std::endl
		<< "		__local uint* reduce_buffer," << std::endl
		<< "		uint shift," << std::endl
		<< "		__global uint* global_histogram)" << std::endl
		<< "	{" << std::endl
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
		<< "		uint start," << std::endl
		<< "		uint N," << std::endl
		<< "		uint loopN," << std::endl
		<< "		__local uint* reduce_buffer," << std::endl
		<< "		uint shift," << std::endl
		<< "		uint first_digit," << std::endl
		<< "		uint last_digit," << std::endl
		<< "		__global uint* global_histogram," << std::endl
		<< "		__global uint* out_offsets, " << std::endl
		<< "		__global VALUE_TYPE* out )" << std::endl
		<< "	{" << std::endl
		<< "        for (int id = get_global_id(0); id < start; id+= get_global_size(0))" << std::endl
		<< "		       {                                                            " << std::endl // clone original vector up to start
		<< "		          out[id] = in[id];                                            " << std::endl // clone original vector up to start
		<< "		          out_offsets[id] = in_offsets[id];                            " << std::endl // clone original vector up to start
		<< "		       }				                                            " << std::endl
		<< "		for (int id = get_global_id(0); id < loopN; id += get_global_size(0))" << std::endl // rescatter rest
		<< "		{" << std::endl
		<< "			uint digit = select((uint)0xF, extract_char(in[id+start], MASK, MASK_SIZE * shift), id+start < N);" << std::endl
		<< "			create_scans(digit, start, N, reduce_buffer, shift,first_digit, last_digit);" << std::endl
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

		<< "}" << std::endl;

	fwrite(str.str().c_str(), str.str().length(), 1, tmp);
	fclose(tmp);
	return str.str();
}

void clearme()
{
	std::ofstream fos("exec.log", std::ofstream::out | std::ofstream::trunc);
	fos << std::endl << "--------------------------------" << std::endl;
	fos.close();
}

template <typename T>
void _print_vector(const viennacl::vector<T>&v, int size)
{
	std::ofstream fos("exec.log", std::ofstream::out | std::ofstream::app);
	fos << std::endl << "--------------------------------" << std::endl;
	for (int i = 0; i < size; ++i)
	{
		fos << " " /*<< std::hex*/ << v(i);
	}
	fos << std::endl << "--------------------------------" << std::endl;
	fos.close();
}

template <typename T>
void _print_vector(const viennacl::vector<T>&v, int start, int end)
{
	std::ofstream fos("exec.log", std::ofstream::out | std::ofstream::app);
	fos << std::endl << "--------------------------------" << std::endl;
	for (int i = start; i < end; ++i)
	{
		fos << " " /*<< std::hex*/ << v(i);
	}
	fos << std::endl << "--------------------------------" << std::endl;
}

/*
Select N smallest values from basic_type vector
*/
template<typename basic_type>
viennacl::vector<unsigned int> radix_select(int N, viennacl::vector<basic_type>& in)
{
	int selectN = N;
	viennacl::vector<unsigned int> src(in.size(), viennacl::traits::context(in));
	viennacl::vector<unsigned int> dst(in.size(), viennacl::traits::context(in));
	viennacl::vector<basic_type> tmp(in.size(), viennacl::traits::context(in));
	// load kernels
	static bool init = false;
	static int num_gpu_groups;
	static int wg_size = 128;
	static int num_digits = 16;
	
	if (!init)
	{
		std::string program_text = generate_kernel<basic_type>();
		
		viennacl::context& the_context = viennacl::traits::context(in);
		viennacl::ocl::context& ctx = const_cast<viennacl::ocl::context&>(the_context.opencl_context());

		std::cout << "Device " << ctx.current_device().name() << std::endl;
		ctx.build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
		ctx.add_program(program_text, std::string("radix_select"));
		num_gpu_groups = ctx.current_device().max_compute_units() * 4 + 1;

		init = true;
	}

	int num_groups = std::min(num_gpu_groups, (int)(in.size() / wg_size) + 1);

	viennacl::ocl::kernel scan_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "scan_digits");
	viennacl::ocl::kernel scatter_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "scatter_digits");
	viennacl::ocl::kernel init_offsets_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "init_offsets");

	scan_kernel.local_work_size(0, wg_size);
	scan_kernel.global_work_size(0, wg_size * num_groups);

	scatter_kernel.local_work_size(0, wg_size);
	scatter_kernel.global_work_size(0, wg_size * num_groups);

	init_offsets_kernel.local_work_size(0, wg_size);
	init_offsets_kernel.global_work_size(0, wg_size* num_groups);
	cl_uint size = src.size();
	viennacl::ocl::enqueue(init_offsets_kernel(size, src));

	int position = 0;
	viennacl::vector<unsigned int> result(N, viennacl::traits::context(in));

	viennacl::vector<unsigned int> global_histogram((0xF + 1) * num_groups, viennacl::traits::context(in));
	viennacl::vector<unsigned int> global_histogram_prefix((0xF + 1) * num_groups + 1, viennacl::traits::context(in));
	viennacl::vector<unsigned int> global_histogram_prefix_2((0xF + 1) * num_groups + 1, viennacl::traits::context(in));
	viennacl::vector<unsigned int> global_histogram_carries(wg_size, viennacl::traits::context(in));
	std::vector< unsigned int > global_histogram_cpu((0xF + 1) * num_groups + 1);
	int shift = 0;// sizeof(basic_type) * 2 - 1;
	int scan_start = 0;
	int scan_end = in.size();
	for (; shift >= 0; --shift)
	{
		int main = ((scan_end - scan_start) / wg_size) * wg_size; // floor to multiple wg size
		int loop_end = main == scan_end ? main : main + wg_size; // add wg size if needed

		viennacl::ocl::enqueue(scan_kernel(in,
			src,
			scan_start,
			scan_end,
			viennacl::ocl::local_mem(sizeof(cl_uint) *wg_size),
			shift,
			global_histogram, 
			global_histogram_prefix, 
			global_histogram_carries
			));

		viennacl::linalg::exclusive_scan(global_histogram, global_histogram_prefix_2);

		_print_vector(global_histogram, global_histogram.size());
		_print_vector(global_histogram_prefix, global_histogram_prefix.size());
		_print_vector(global_histogram_prefix_2, global_histogram_prefix_2.size());


		int digit = 1;
		viennacl::copy(global_histogram_prefix, global_histogram_cpu);
		global_histogram_cpu[global_histogram_cpu.size() - 1] = global_histogram_cpu[global_histogram_cpu.size() - 2]; // fix last element
		for (; digit < num_digits; ++digit)
		{
			int offset = global_histogram_cpu[num_groups * digit];
			if (offset >= N)
				break;
		}

		int hist_max = global_histogram_cpu[num_groups * digit];
		int hist_min = global_histogram_cpu[num_groups * (digit - 1)];

		if (hist_min == 0 && (digit < num_digits - 1 && hist_max == global_histogram_cpu[num_groups * (digit + 1)]))
			continue;

		//viennacl::copy(in.begin(), in.begin() + scan_start, tmp.begin());

		viennacl::ocl::enqueue(scatter_kernel(
			in,
			src,
			scan_start,
			scan_end,
			loop_end,
			viennacl::ocl::local_mem(sizeof(cl_uint) *wg_size * (0xF + 1)),
			shift,
			0,
			digit,
			global_histogram_prefix,
			dst,
			tmp
			));

		/*	std::cout << "Result vector before:"  << scan_end << std::endl;
		_print_vector(in, scan_end);
		std::cout << "Result vector after:" << std::endl;
		_print_vector(tmp, scan_end);
		std::cout << "Scatter : " << std::endl;
		_print_vector(dst, scan_end);
		*/
		tmp.fast_swap(in);
		dst.fast_swap(src);

		if (hist_max == N)
			break;
		if (hist_max> N && hist_min < N)
		{
			if (hist_min > 0)
			{
				N -= (hist_min - scan_start);
				if (N == 0)
					break;
			}
			scan_start = hist_min;
			scan_end = hist_max;
		}


	}
	src.resize(selectN, true);
	return src;
}



}


template <typename T>
static void print_vector(const viennacl::vector<T>&v, std::string term = "", bool dec = true, int step =1)
{
	std::ofstream fos("exec.log", std::ofstream::out | std::ofstream::app);
	fos << term << " ";
	for (int i = 0; i < v.size(); i += step)
	{
		fos << " ";
		if (!dec)
			fos << std::hex;
		fos << v(i);
	}
	fos << std::endl;
	fos.close();
}

template <typename T>
static void print_vector(const viennacl::vector_range<T>&v, std::string term = "", bool dec = true, int step = 1)
{
	std::ofstream fos("exec.log", std::ofstream::out | std::ofstream::app);
	fos << term << " ";
	for (int i = 0; i < v.size(); i += step)
	{
		fos << " ";
		if (!dec)
			fos << std::hex;
		fos << v(i);
	}
	fos << std::endl;
	fos.close();
}

template <typename T>
static void print_vector_std(const std::vector<T>&v, std::string term = "", bool dec = true, int step = 1)
{
	std::ofstream fos("exec.log", std::ofstream::out | std::ofstream::app);
	fos << term << " ";
	for (auto it = v.begin(); it != v.end(); it = it + step)
	{ 
		fos << " ";
		if (!dec)
			fos << std::hex;
		fos << *it;
	}
	fos << std::endl;
	fos.close();
}

template <typename T>
static void print_vector_selected(const viennacl::vector<T>&v, const viennacl::vector<unsigned int>& selected)
{
	std::ofstream fos("exec.log", std::ofstream::out | std::ofstream::app);
	std::vector<T> test;
	for (auto idx : selected)
	{
		test.push_back(v(idx));
	}
	std::stable_sort(test.begin(), test.end());
	for (auto val : test)
		fos << " " << val;
	fos << std::endl;
	fos.close();
}


TEST(radix_sort, _third_stage)
{
	using namespace magic_hamster;
	clearme();

	viennacl::ocl::current_context().cache_path("c:/tmp/");
	static bool init = false;
	static int num_gpu_groups;
	static int wg_size = 4;
	static int num_digits = 16;

	viennacl::backend::mem_handle opencl_carries;
	viennacl::backend::memory_create(opencl_carries, sizeof(cl_uint) * 128, viennacl::ocl::current_context());

	if (!init)
	{
		std::string program_text = generate_kernel<unsigned int>();


		viennacl::ocl::context& ctx = viennacl::ocl::current_context();

		std::cout << "Device " << ctx.current_device().name() << std::endl;
		ctx.build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
		ctx.add_program(program_text, std::string("radix_select"));
		num_gpu_groups = ctx.current_device().max_compute_units() * 4 + 1;

		init = true;
	}

	num_gpu_groups = 128;

	viennacl::ocl::current_context().add_device_queue(viennacl::ocl::current_context().current_device().id());
	viennacl::ocl::kernel scan_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "scan_digits");
	viennacl::ocl::kernel scatter_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "scatter_digits");
	viennacl::ocl::kernel scan_with_offset = viennacl::ocl::current_context().get_kernel("radix_select", "scan_with_offset");

	scan_kernel.local_work_size(0, wg_size);
	scan_kernel.global_work_size(0, wg_size * num_gpu_groups);

	scatter_kernel.local_work_size(0, wg_size);
	scatter_kernel.global_work_size(0, wg_size * num_gpu_groups);

	viennacl::vector<unsigned int> global_histogram_prefix((0xF + 1) * num_gpu_groups + 1, viennacl::ocl::current_context());
	viennacl::vector<unsigned int> secondary_histogram_prefix((0xF + 1) * num_gpu_groups + 1, viennacl::ocl::current_context());
	viennacl::vector<unsigned int> global_histogram_carries(128, viennacl::ocl::current_context());

	int max_num_digits = 20;

	viennacl::vector<unsigned int> src(128 * 16*   max_num_digits, viennacl::ocl::current_context());
	viennacl::vector<unsigned int> dst(128 * 16 * max_num_digits, viennacl::ocl::current_context());

	viennacl::vector<unsigned int> scan_context(128, viennacl::ocl::current_context());
	viennacl::vector<int> input(16 * max_num_digits, viennacl::ocl::current_context());
	viennacl::vector<int> output(16 * max_num_digits, viennacl::ocl::current_context());

	int count = 16 * max_num_digits;
	for (int i = 0; i < max_num_digits; i++)
		for (int j = 0; j < 16; ++j)
			input(i * 16 + j) = j *0xF + i;
	std::vector<int> test(input.size());
	viennacl::copy(input, test);

	int shift = sizeof(int) * 2 - 1;
	shift = 1;

	int start = 0;
	int end = input.size();
	int N = 60;
	print_vector(input, "input_before", false);
	viennacl::ocl::enqueue(
			scan_kernel(
			N,
			input,
			src,
			output,
			dst,
			end,
			shift,
			global_histogram_prefix,
			global_histogram_carries, 
			scan_context
			)
			);
		
		/*viennacl::ocl::enqueue(
			scan_with_offset(input,
				src,
				output,
				dst,
				global_histogram_prefix,
				global_histogram_carries,
				scan_context)
		);*/
	

		std::vector<int> raw(input.size());
		viennacl::copy(input, raw);
//	print_vector_std(test, "lol_selection", false);
		std::sort(test.begin(), test.end());
		test.resize(N);
		print_vector_std(test, "test_after", false);
		test.clear();
		test.resize(N);
		
		print_vector(input, "raw_input", false);
		print_vector(dst, "out_offset", false);
		print_vector(scan_context, "ctx");
		viennacl::copy(input.begin(), input.begin() + N, test.begin());
		std::sort(test.begin(), test.end());
		print_vector_std(test, "input_after", false);
		print_vector(global_histogram_prefix, "histogram", false);

		std::vector<unsigned int>  test_ctx(scan_context.size());
		viennacl::copy(scan_context.begin(), scan_context.end(), test_ctx.begin());


		// do next step 
		shift = 0;

		/*
		struct scatter_context
		{
			uint hist_min;0
			uint hist_max;1
			uint stop;2
			uint min_digit;3
			uint max_digit;4
			uint shift;5
			uint debug_n;6
			uint debug_start;7
			uint debug_end;8
		};*/
		int shift_pre = scan_context(5);
		int subN = scan_context(6);
		start = scan_context(7);
		end = scan_context(8);
		viennacl::ocl::enqueue(
			scan_with_offset(
				input,
				src,
				output,
				dst,
				global_histogram_prefix,
				global_histogram_carries,
				scan_context
				)
			);
		print_vector(input, "result", false);
		
		printf("");


}

TEST(radix_sort, DISABLED_second_stage)
{
	using namespace magic_hamster;
	clearme();

	viennacl::ocl::current_context().cache_path("c:/tmp/");
	static bool init = false;
	static int num_gpu_groups;
	static int wg_size = 128;
	static int num_digits = 16;
	viennacl::backend::mem_handle opencl_carries;
	viennacl::backend::memory_create(opencl_carries, sizeof(cl_uint) * 128, viennacl::ocl::current_context());

	if (!init)
	{
		std::string program_text = generate_kernel<unsigned int>();

		
		viennacl::ocl::context& ctx = viennacl::ocl::current_context();

		std::cout << "Device " << ctx.current_device().name() << std::endl;
		ctx.build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
		ctx.add_program(program_text, std::string("radix_select"));
		num_gpu_groups = ctx.current_device().max_compute_units() * 4 + 1;

		init = true;
	}

	num_gpu_groups = 2;

	viennacl::ocl::current_context().add_device_queue(viennacl::ocl::current_context().current_device().id());
	viennacl::ocl::kernel scan_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "scan_digits");
	viennacl::ocl::kernel scatter_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "scatter_digits");


	scan_kernel.local_work_size(0, wg_size);
	scan_kernel.global_work_size(0, wg_size * num_gpu_groups);

	scatter_kernel.local_work_size(0, wg_size);
	scatter_kernel.global_work_size(0, wg_size * num_gpu_groups);

	viennacl::vector<unsigned int> global_histogram((0xF + 1) * num_gpu_groups, viennacl::ocl::current_context());
	viennacl::vector<unsigned int> global_histogram_prefix((0xF + 1) * num_gpu_groups + 1, viennacl::ocl::current_context());
	viennacl::vector<unsigned int> global_histogram_carries(128, viennacl::ocl::current_context());

	int max_num_digits = 4;

	viennacl::vector<unsigned int> src(16 * max_num_digits, viennacl::ocl::current_context());
	viennacl::vector<unsigned int> dst(16 * max_num_digits, viennacl::ocl::current_context());

	viennacl::vector<unsigned int> context(4, viennacl::ocl::current_context());
	viennacl::vector<int> input(16 * max_num_digits, viennacl::ocl::current_context());
	viennacl::vector<int> output(16 * max_num_digits, viennacl::ocl::current_context());

	int count = 16 * max_num_digits;
	for (int i = 0; i < max_num_digits; i++)
		for (int j = 0; j < 16; ++j)
			input(i * 16 + j) = (--count);

	int shift = sizeof(int) * 2 - 1;

	int start = 0;
	int end = input.size();
	int N = 30;
	for (; shift >= 0; --shift)
	{
		viennacl::ocl::enqueue(
			scan_kernel(input,
			src,
			output,
			dst,
			start,
			end,
			viennacl::ocl::local_mem(sizeof(cl_uint) *wg_size),
			shift, // shift
			global_histogram,
			global_histogram_prefix,
			global_histogram_carries
			)
			);
		print_vector(input, "input_");
		print_vector(global_histogram_prefix, "prefix");
		
		viennacl::ocl::enqueue(
			scatter_kernel(N, input, src, output, dst, start, end, shift, viennacl::ocl::local_mem(sizeof(cl_uint) *wg_size), global_histogram_prefix, context)
			);
		if (context(2) != 1)
		{
			start = context(0);
			end = context(1);
			if (end > N)
				N = N - start;
			else
				shift = -1; // break out from selection
			if (end > input.size())
				end = input.size();
		}
		
		print_vector(input, "input_upd");
		print_vector(output, "scatter");

	}
	printf("");
}



TEST(radix_sort, DISABLED_first_stage)
{
	using namespace magic_hamster;
	clearme();

	viennacl::ocl::current_context().cache_path("c:/tmp/");
	static bool init = false;
	static int num_gpu_groups;
	static int wg_size = 128;
	static int num_digits = 16;
	viennacl::vector<unsigned int> in(viennacl::scalar_vector<unsigned int>(1000, 1, viennacl::ocl::current_context()));
	viennacl::vector<unsigned int> out(viennacl::scalar_vector<unsigned int>(1001, 1, viennacl::ocl::current_context()));
	viennacl::vector<unsigned int> out2(viennacl::scalar_vector<unsigned int>(1001, 1, viennacl::ocl::current_context()));
	viennacl::backend::mem_handle opencl_carries;
	viennacl::backend::memory_create(opencl_carries, sizeof(cl_uint) * 128, viennacl::ocl::current_context());

	if (!init)
	{
		std::string program_text = generate_kernel<unsigned int>();

		viennacl::context& the_context = viennacl::traits::context(in);
		viennacl::ocl::context& ctx = const_cast<viennacl::ocl::context&>(the_context.opencl_context());

		std::cout << "Device " << ctx.current_device().name() << std::endl;
		ctx.build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
		ctx.add_program(program_text, std::string("radix_select"));
		num_gpu_groups = ctx.current_device().max_compute_units() * 4 + 1;

		init = true;
	}

	num_gpu_groups = 1;

	viennacl::ocl::current_context().add_device_queue(viennacl::ocl::current_context().current_device().id());
	viennacl::ocl::kernel scan_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "scan_digits");
	viennacl::ocl::kernel scatter_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "scatter_digits");
	viennacl::ocl::kernel find_limits_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "find_limits");
	
	
	scan_kernel.local_work_size(0, wg_size);
	scan_kernel.global_work_size(0, wg_size * num_gpu_groups);

	scatter_kernel.local_work_size(0, wg_size);
	scatter_kernel.global_work_size(0, wg_size * num_gpu_groups);

	viennacl::vector<unsigned int> global_histogram((0xF + 1) * num_gpu_groups, viennacl::traits::context(in));
	viennacl::vector<unsigned int> global_histogram_prefix((0xF + 1) * num_gpu_groups + 1, viennacl::traits::context(in));
	viennacl::vector<unsigned int> global_histogram_carries(128, viennacl::traits::context(in));

	int max_num_digits = 1;

	viennacl::vector<unsigned int> src(16 * max_num_digits, viennacl::ocl::current_context());
	viennacl::vector<unsigned int> dst(16 * max_num_digits, viennacl::ocl::current_context());

	viennacl::scalar<unsigned int> stage(0, viennacl::ocl::current_context());
	viennacl::vector<int> input(16 * max_num_digits, viennacl::ocl::current_context());
	viennacl::vector<int> output(16 * max_num_digits, viennacl::ocl::current_context());
	
	for (int i = 0; i < max_num_digits; i++)
		for (int j = 0; j < 16; ++j)
			input(i * 16 + j) = j;

	int start = 0;
	int N = input.size();

	viennacl::ocl::enqueue(
		scan_kernel(input, 
		src,
		output,
		dst,
		stage,
		0,
		N,
		viennacl::ocl::local_mem(sizeof(cl_uint) *wg_size),
		(unsigned int)0, // shift
		global_histogram,
		global_histogram_prefix,
		global_histogram_carries
		)
		);

	print_vector(input);
	print_vector(src);
	print_vector(global_histogram);
	print_vector(global_histogram_prefix);
	print_vector(global_histogram_carries);
	std::vector<unsigned int> global_histogram_prefix_cpu(global_histogram_prefix.size());
	viennacl::copy(global_histogram_prefix, global_histogram_prefix_cpu);

	viennacl::scalar<unsigned int> hist_min(0, viennacl::ocl::current_context());
	viennacl::scalar<unsigned int> hist_max(0, viennacl::ocl::current_context());
	find_limits_kernel.local_work_size(0, 128);
	find_limits_kernel.global_work_size(0, 128);

	int num_groups = 10;

	viennacl::vector<unsigned int> idx(128, viennacl::traits::context(in)), idx_next(128, viennacl::traits::context(in));
	viennacl::vector<unsigned int> prefix(num_groups * (0xF + 1)+1, viennacl::traits::context(in));
	
	for (int i = 0; i < 0xF + 1; ++i)
	{
		for (int g = 0; g < num_groups; ++g)
		{
			int idx = i*num_groups + g;
			prefix(idx) = i;
		}
	}
	prefix(num_groups * (0xF + 1)) = prefix(num_groups * (0xF + 1) - 1) + 1;
	print_vector(prefix);
	viennacl::ocl::enqueue(
		find_limits_kernel(3, prefix, num_groups, hist_min, hist_max)
		);

	print_vector(idx);
	print_vector(idx_next);

	int hist_min_d = hist_min;
	int hist_max_d = hist_max;
	ASSERT_EQ(hist_min_d, 2);
	ASSERT_EQ(hist_max_d, 3);

	viennacl::ocl::enqueue(
		find_limits_kernel(15, prefix, num_groups, hist_min, hist_max)
		);

	ASSERT_EQ(hist_min_d, 14);
	ASSERT_EQ(hist_max_d, 15);

	viennacl::ocl::enqueue(
		scatter_kernel(in, src, out, dst, stage, 0, N, viennacl::ocl::local_mem(sizeof(cl_uint) *wg_size), global_histogram_prefix)
		);
	
	print_vector(dst);
	print_vector(out);

	printf("");
}

TEST(radix_sort, DISABLED_bench_scans)
{
	using namespace magic_hamster;
	clearme();
	
	viennacl::ocl::current_context().cache_path("c:/tmp/");
	static bool init = false;
	static int num_gpu_groups;
	static int wg_size = 128;
	static int num_digits = 16;
	viennacl::vector<unsigned int> in(viennacl::scalar_vector<unsigned int>(1000, 1, viennacl::ocl::current_context()));
	viennacl::vector<unsigned int> out(viennacl::scalar_vector<unsigned int>(1001, 1, viennacl::ocl::current_context()));
	viennacl::vector<unsigned int> out2(viennacl::scalar_vector<unsigned int>(1001, 1, viennacl::ocl::current_context()));
	viennacl::backend::mem_handle opencl_carries;
	viennacl::backend::memory_create(opencl_carries, sizeof(cl_uint) * 128, viennacl::ocl::current_context());

	if (!init)
	{
		std::string program_text = generate_kernel<unsigned int>();

		viennacl::context& the_context = viennacl::traits::context(in);
		viennacl::ocl::context& ctx = const_cast<viennacl::ocl::context&>(the_context.opencl_context());

		std::cout << "Device " << ctx.current_device().name() << std::endl;
		ctx.build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
		ctx.add_program(program_text, std::string("radix_select"));
		num_gpu_groups = ctx.current_device().max_compute_units() * 4 + 1;

		init = true;
	}

	viennacl::ocl::current_context().add_device_queue(viennacl::ocl::current_context().current_device().id());
	viennacl::ocl::kernel scan_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "scan_1");
	scan_kernel.local_work_size(0, 128);
	scan_kernel.global_work_size(0, 128 * 128);
	int size = in.size();
	for (int k = 0; k < 1000; ++k)
	{

	
	{

		using namespace std::chrono;
		viennacl::context& the_context = viennacl::traits::context(in);
		viennacl::ocl::context& ctx = const_cast<viennacl::ocl::context&>(the_context.opencl_context());


		steady_clock::time_point t1 = steady_clock::now();
		for (int a = 0; a < 100; a ++ )
		{
			for (int i = 0; i < 1000; ++i)
				viennacl::linalg::exclusive_scan(in, out2);
			ctx.get_queue().finish();

		}

		steady_clock::time_point t2 = steady_clock::now();

		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		std::cout << "HOLY COW " << time_span.count() << std::endl;
	}

	{
		
		using namespace std::chrono;
		viennacl::context& the_context = viennacl::traits::context(in);
		viennacl::ocl::context& ctx  = const_cast<viennacl::ocl::context&>(the_context.opencl_context());


		steady_clock::time_point t1 = steady_clock::now();
		for (int a = 0; a < 1000; a++)
		{
			for (int i = 0; i < 100; ++i)
				viennacl::ocl::enqueue(scan_kernel(in, size + 1, out, opencl_carries.opencl_handle()));
			ctx.get_queue().finish();
		}

		steady_clock::time_point t2 = steady_clock::now();

		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		std::cout << "DEVICE STUFF " << time_span.count() << std::endl;
	}

	}
	printf("");
}

TEST(radix_sort, DISABLED_inclusive_scan_test)
{
	using namespace magic_hamster;
	clearme();
	viennacl::ocl::current_context().cache_path("c:/tmp/");
	static bool init = false;
	static int num_gpu_groups;
	static int wg_size = 128;
	static int num_digits = 16;
	viennacl::vector<unsigned int> in(viennacl::scalar_vector<unsigned int>(1000,1, viennacl::ocl::current_context()));
	viennacl::vector<unsigned int> out(viennacl::scalar_vector<unsigned int>(1001, 1, viennacl::ocl::current_context()));
	viennacl::vector<unsigned int> out2(viennacl::scalar_vector<unsigned int>(1001, 1, viennacl::ocl::current_context()));
	viennacl::vector<unsigned int> carries(viennacl::scalar_vector<unsigned int>(128, 1, viennacl::ocl::current_context()));
	if (!init)
	{
		std::string program_text = generate_kernel<unsigned int>();

		viennacl::context& the_context = viennacl::traits::context(in);
		viennacl::ocl::context& ctx = const_cast<viennacl::ocl::context&>(the_context.opencl_context());

		std::cout << "Device " << ctx.current_device().name() << std::endl;
		ctx.build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
		ctx.add_program(program_text, std::string("radix_select"));
		num_gpu_groups = ctx.current_device().max_compute_units() * 4 + 1;

		init = true;
	}

	viennacl::ocl::current_context().add_device_queue(viennacl::ocl::current_context().current_device().id());
	viennacl::ocl::kernel scan_kernel = viennacl::ocl::current_context().get_kernel("radix_select", "scan_1");
	scan_kernel.local_work_size(0, 128);
	scan_kernel.global_work_size(0, 128*128);
	int size = in.size();
	viennacl::ocl::enqueue(scan_kernel(in, size+1 , out, carries));

	viennacl::linalg::inclusive_scan(in, out2);

	_print_vector(in, in.size());
	_print_vector(out, out.size());
	_print_vector(out2, out2.size());
	_print_vector(carries, carries.size());
	printf("");
}


TEST(radix_sort, DISABLED_sort_test_lol)
{
	using namespace magic_hamster;
	clearme();
	viennacl::ocl::current_context().cache_path("c:/tmp/");
	viennacl::ocl::current_context().add_device_queue(viennacl::ocl::current_context().current_device().id());
	int size = 1024;
	viennacl::vector<unsigned int> selection(size, viennacl::ocl::current_context());
	for (int i = 0; i < size; i++)
		selection(i) = size - i;
	print_vector(selection);
	viennacl::vector<unsigned int> offsets = magic_hamster::radix_select<uint32_t>(32, selection);
	print_vector_selected(selection, offsets);
	std::vector<unsigned int> result(offsets.size());
	viennacl::copy(offsets, result);
	
	//ASSERT_EQ(offsets.size(), 32);
	for (int i = 0; i < 32; ++i)
	{
		ASSERT_EQ(result[i], i + 1);
	}
}


#endif