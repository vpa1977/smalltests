#include <vector>
#include "gtest/gtest.h"

#include "viennacl/context.hpp"
#include "viennacl/ml/knn.hpp"

#include <boost/numeric/ublas/matrix.hpp>
#include <atomic>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

/*
 * sort.cpp
 *
 *  Created on: 15/06/2015
 *      Author: bsp
 */

// debug sorting kernsl
using namespace std::chrono;

// NB. 6% speedup from shuffle. cost ?
#define NAME "bitonic_sort_int.cl"
#define TYPE int

TEST(test_sort, DISABLED_cas_compare_scalar)
{
	FILE * tmp = fopen(NAME, "rb");
	fseek(tmp,0,SEEK_END);
	std::vector<char> binary;
	binary.resize(ftell(tmp));
	rewind(tmp);
	fread(&binary[0], binary.size(), 1, tmp);
	fclose(tmp);
	binary.push_back(0);
	static viennacl::context g_context =   viennacl::ocl::current_context();
	static bool init = false;
	viennacl::ocl::context* ctx = g_context.opencl_pcontext();
	//ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
	std::string program_text(&binary[0]);
	ctx->add_program(program_text, std::string("test"));


#define LEN 128*128
	std::vector<TYPE> holy_cow;
	for (int i = 0; i < LEN; ++i)
		holy_cow.push_back(i);

	viennacl::vector<TYPE> vcl(LEN, g_context);
	viennacl::copy(holy_cow, vcl);


	viennacl::ocl::kernel scalar_kernel =  ctx->get_kernel("test", "scalar_test");
	viennacl::ocl::kernel shuffle_kernel =  ctx->get_kernel("test", "shuffle_test");

	scalar_kernel.local_work_size(0, std::min(256, LEN/2));
	scalar_kernel.global_work_size(0, LEN/2);

	shuffle_kernel.local_work_size(0, std::min(256, LEN/4));
	shuffle_kernel.global_work_size(0, LEN/4);


	double cnt = 0;

	for (int iter = 0 ; iter < 30 ; ++iter )
	{
		steady_clock::time_point t1 = steady_clock::now();

		viennacl::ocl::context* ctx = g_context.opencl_pcontext();
		for (int i = 0; i < 10000 ; ++i)
		{
			viennacl::ocl::enqueue(scalar_kernel(vcl));
		}
		ctx->get_queue().finish();

		steady_clock::time_point t2 = steady_clock::now();

		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		cnt += time_span.count();
	}
	cnt = cnt/30;
	std::cout << "Scalar time " << cnt << std::endl ;
}


TEST(test_sort, DISABLED_cas_compare_shuffle)
{
	FILE * tmp = fopen(NAME, "rb");
	fseek(tmp,0,SEEK_END);
	std::vector<char> binary;
	binary.resize(ftell(tmp));
	rewind(tmp);
	fread(&binary[0], binary.size(), 1, tmp);
	fclose(tmp);
	binary.push_back(0);
	static viennacl::context g_context =   viennacl::ocl::current_context();
	static bool init = false;
	viennacl::ocl::context* ctx = g_context.opencl_pcontext();
	//ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
	std::string program_text(&binary[0]);
	ctx->add_program(program_text, std::string("test"));


#define LEN 128*128
	std::vector<TYPE> holy_cow;
	for (int i = 0; i < LEN; ++i)
		holy_cow.push_back(i);

	viennacl::vector<TYPE> vcl(LEN, g_context);
	viennacl::copy(holy_cow, vcl);


	viennacl::ocl::kernel scalar_kernel =  ctx->get_kernel("test", "scalar_test");
	viennacl::ocl::kernel shuffle_kernel =  ctx->get_kernel("test", "shuffle_test");

	scalar_kernel.local_work_size(0, std::min(256, LEN/2));
	scalar_kernel.global_work_size(0, LEN/2);

	shuffle_kernel.local_work_size(0, std::min(256, LEN/4));
	shuffle_kernel.global_work_size(0, LEN/4);


	double cnt = 0;

	cnt = 0;
	for (int iter = 0 ; iter < 30 ; ++iter )
	{
		steady_clock::time_point t1 = steady_clock::now();

		viennacl::ocl::context* ctx = g_context.opencl_pcontext();
		for (int i = 0; i < 10000 ; ++i)
		{
			viennacl::ocl::enqueue(shuffle_kernel(vcl));
		}
		ctx->get_queue().finish();

		steady_clock::time_point t2 = steady_clock::now();

		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		cnt += time_span.count();
	}
	cnt = cnt/30;
	std::cout << "Shuffle time " << cnt << std::endl ;
}


TEST(radix_sort, DISABLED_convert_kernel)
{
	FILE * tmp = fopen("convert.cl", "rb");
	fseek(tmp,0,SEEK_END);
	std::vector<char> binary;
	binary.resize(ftell(tmp));
	rewind(tmp);
	fread(&binary[0], binary.size(), 1, tmp);
	fclose(tmp);
	binary.push_back(0);
	static viennacl::context g_context =   viennacl::ocl::current_context();
	static bool init = false;
	viennacl::ocl::context* ctx = g_context.opencl_pcontext();
	ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
	std::string program_text(&binary[0]);
	ctx->add_program(program_text, std::string("test"));

	viennacl::vector<double> input(128, g_context);
	viennacl::vector<uint64_t> output(128, g_context);
	for (int i = 0 ;i < 128; ++i)
		input(i) = (double)1 / i;

	viennacl::ocl::kernel convert = ctx->get_kernel("test", "convert");
	convert.local_work_size(0, 128);
	convert.global_work_size(0, 128);
	viennacl::ocl::enqueue( convert( input, output));
	ctx->get_queue().finish();

	for (int i = 0 ;i < 128 ; ++i)
	{
		double val = input(i);
		int64_t * ptr = (int64_t*)&val;
		uint64_t mask = uint64_t (-(*ptr >> 63) | 0x8000000000000000ull );
		uint64_t result = *ptr ^ mask;
		ASSERT_EQ(result, output(i));
	}

}

TEST(radix_sort, DISABLED_workgroup_reduce)
{
	FILE * tmp = fopen("convert.cl", "rb");
	fseek(tmp,0,SEEK_END);
	std::vector<char> binary;
	binary.resize(ftell(tmp));
	rewind(tmp);
	fread(&binary[0], binary.size(), 1, tmp);
	fclose(tmp);
	binary.push_back(0);
	static viennacl::context g_context =   viennacl::ocl::current_context();
	static bool init = false;
	viennacl::ocl::context* ctx = g_context.opencl_pcontext();
	ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
	std::string program_text(&binary[0]);
	ctx->add_program(program_text, std::string("test"));

	viennacl::vector<int> input(256, g_context);

	viennacl::ocl::kernel convert = ctx->get_kernel("test", "workgroup_reduction_test");
	viennacl::ocl::local_mem local_buffer(256);
	int size = 256;
	for (int s = 16; s <=256; s++)
	{
		for (int i = 0 ;i < size; ++i)
			input(i) = 1;
		convert.local_work_size(0,size);
		convert.global_work_size(0,size);
		viennacl::ocl::enqueue( convert( input, local_buffer));
		ctx->get_queue().finish();
		int total = input(0);
		ASSERT_EQ(size, total);
	}


}


TEST(radix_sort, DISABLED_workgroup_scan)
{
	FILE * tmp = fopen("convert.cl", "rb");
	fseek(tmp,0,SEEK_END);
	std::vector<char> binary;
	binary.resize(ftell(tmp));
	rewind(tmp);
	fread(&binary[0], binary.size(), 1, tmp);
	fclose(tmp);
	binary.push_back(0);
	static viennacl::context g_context =   viennacl::ocl::current_context();
	static bool init = false;
	viennacl::ocl::context* ctx = g_context.opencl_pcontext();
	ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
	std::string program_text(&binary[0]);
	ctx->add_program(program_text, std::string("test"));

	viennacl::vector<int> input(256, g_context);

	viennacl::ocl::kernel convert = ctx->get_kernel("test", "workgroup_scan_test");
	viennacl::ocl::local_mem local_buffer(sizeof(cl_uint)*256* 16);
	//for (int size = 16; size <=256; size = size *2)
	int size = 256;
	{
		for (int i = 0 ;i < size; ++i)
			input(i) = 1;
		convert.local_work_size(0,size);
		convert.global_work_size(0,size);
		viennacl::ocl::enqueue( convert( input, local_buffer));
		ctx->get_queue().finish();
		for (int i = 0 ;i < size; ++i)
		{
			ASSERT_EQ(i+1, input(i));
		}
		std::cout << std::endl;

	}


}



TEST(radix_sort, DISABLED_scan_kernel_test)
{
	FILE * tmp = fopen("convert.cl", "rb");
	fseek(tmp,0,SEEK_END);
	std::vector<char> binary;
	binary.resize(ftell(tmp));
	rewind(tmp);
	fread(&binary[0], binary.size(), 1, tmp);
	fclose(tmp);
	binary.push_back(0);
	static viennacl::context g_context =   viennacl::ocl::current_context();
	static bool init = false;
	viennacl::ocl::context* ctx = g_context.opencl_pcontext();
	ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
	std::string program_text(&binary[0]);
	ctx->add_program(program_text, std::string("test"));
	int shift = 15;
	std::vector<uint64_t> sample_data;
	for (int i = 0 ;i < 1000; i++)
		sample_data.push_back(i);

	while (shift >= 0 )
	{
		int counts[16] = {};
		for (int i = 0 ;i < 1000; i++)
		{
			uint64_t val = sample_data[i];
			val = (val >> (4 * shift) ) & 0xF;
			counts[val]++;
		}
		std::cout << "shift " << shift << " ";
		for (int i = 0 ;i < 16 ; ++i)
			std::cout << counts[i] << " ";
		std::cout << std::endl;

		shift--;
	}


	viennacl::ocl::kernel scan_kernel = ctx->get_kernel("test", "scan_digits");

	int N = 1000;
	int num_groups = 1;

	scan_kernel.local_work_size(0,256);
	scan_kernel.global_work_size(0,num_groups*256);

	viennacl::vector<uint64_t> debug(N, g_context);
	viennacl::vector<uint64_t> values(N, g_context);
	viennacl::vector<unsigned int> global_histogram( 16 * num_groups, g_context);
	for (int i = 0 ; i < N ; i++)
		values(i) = i;
	for (int shift = 15; shift >= 0 ; shift --)
	{
		//__global double* in,  uint N, __local uint* reduce_buffer, uint shift, __global uint* global_histogram
		viennacl::ocl::enqueue( scan_kernel(values,0, N,viennacl::ocl::local_mem(256* sizeof(cl_uint)), shift, global_histogram));
		viennacl::vector<unsigned int> global_offsets( 16 * num_groups, viennacl::traits::context(global_histogram));
		std::vector<unsigned int> cpu_global_offsets(16 * num_groups);
		viennacl::linalg::exclusive_scan(global_histogram,global_offsets); // do on CPU ?
		viennacl::copy(global_offsets, cpu_global_offsets);
		std::cout << "shift " << shift << " ";
		for (int i = 1 ;i < 16 ; ++i)
		{
			std::cout << (cpu_global_offsets[i*num_groups] - cpu_global_offsets[(i-1)*num_groups]) << " ";
		}
		std::cout << std::endl;

	}
}



TEST(radix_sort, simple_scatter_test)
{
	FILE * tmp = fopen("convert.cl", "rb");
	fseek(tmp,0,SEEK_END);
	std::vector<char> binary;
	binary.resize(ftell(tmp));
	rewind(tmp);
	fread(&binary[0], binary.size(), 1, tmp);
	fclose(tmp);
	binary.push_back(0);
	static viennacl::context g_context =   viennacl::ocl::current_context();
	static bool init = false;
	viennacl::ocl::context* ctx = g_context.opencl_pcontext();
	ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
	std::string program_text(&binary[0]);
	ctx->add_program(program_text, std::string("test"));
	int shift = 15;

	viennacl::ocl::kernel scan_kernel = ctx->get_kernel("test", "scan_digits");
	viennacl::ocl::kernel scatter_kernel = ctx->get_kernel("test", "scatter_digits");

	int N = 1000;
	int num_groups = 1;

	scan_kernel.local_work_size(0,256);
	scan_kernel.global_work_size(0,num_groups*256);

	scatter_kernel.local_work_size(0,256);
	scatter_kernel.global_work_size(0,num_groups*256);

	viennacl::vector<uint64_t> debug(N, g_context);
	viennacl::vector<uint64_t> values(N, g_context), scatter_values(N, g_context);
	viennacl::vector<unsigned int> keys_in(N, g_context), keys_out(N,g_context);
	std::vector<uint64_t> trace(N);
	viennacl::vector<unsigned int> global_histogram( 16 * num_groups, g_context);
	for (int i = 0 ; i < N ; i++)
		values(i) = N-i;
	for (int shift = 2; shift >= 0 ; shift --)
	{
		//__global double* in,  uint N, __local uint* reduce_buffer, uint shift, __global uint* global_histogram
		viennacl::ocl::enqueue( scan_kernel(values,0, N,viennacl::ocl::local_mem(256* sizeof(cl_uint)), shift, global_histogram));
		viennacl::vector<unsigned int> global_offsets( 16 * num_groups, viennacl::traits::context(global_histogram));
		std::vector<unsigned int> cpu_global_offsets(16 * num_groups);
		viennacl::linalg::exclusive_scan(global_histogram,global_offsets); // do on CPU ?
		viennacl::copy(global_offsets, cpu_global_offsets);

		std::cout << "shift " << shift << " ";
		for (int i = 1 ;i < 16 ; ++i)
		{
			std::cout << (cpu_global_offsets[i*num_groups] - cpu_global_offsets[(i-1)*num_groups]) << " ";
		}
		std::cout << std::endl;

		viennacl::ocl::enqueue( scatter_kernel(
					viennacl::ocl::local_mem(sizeof(cl_uint)  * 256),
					values,
					scatter_values,
					global_offsets,
					0,
					N,
					0,
					0,
					shift,
					keys_in,
					keys_out
					));

		viennacl::copy(scatter_values, trace);
		for (int i = 0 ;i< trace.size() ; ++i)
		{
			std::cout << i<< "=" << trace[i] << "(" << keys_out(i) << ")" <<std::endl;
		}
		std::cout << std::endl;

	}
}




void sample_select(viennacl::vector<uint64_t>& in, int K, viennacl::vector<uint64_t>& out)
{
	std::vector<double> trace(1000);
	static bool init = false;
	viennacl::ocl::context* ctx = viennacl::traits::context(in).opencl_pcontext();
	if (!init)
	{
		init = true;
		FILE * tmp = fopen("convert.cl", "rb");
		fseek(tmp,0,SEEK_END);
		std::vector<char> binary;
		binary.resize(ftell(tmp));
		rewind(tmp);
		fread(&binary[0], binary.size(), 1, tmp);
		fclose(tmp);
		binary.push_back(0);

		ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
		std::string program_text(&binary[0]);
		ctx->add_program(program_text, std::string("test"));
	}
	viennacl::ocl::kernel& scan_kernel = ctx->get_kernel("test", "scan_digits");
	viennacl::ocl::kernel& scatter_kernel = ctx->get_kernel("test", "scatter_digits");
	int num_groups = 1;//4*ctx->current_device().max_compute_units() +1;

	scan_kernel.local_work_size(0,256);
	scan_kernel.global_work_size(0,num_groups*256);

	scatter_kernel.local_work_size(0,256);
	scatter_kernel.global_work_size(0,num_groups*256);

	viennacl::vector<unsigned int> keys_in(in.size(), viennacl::traits::context(in)),keys_out(in.size(), viennacl::traits::context(in));
	viennacl::vector<unsigned int> global_histogram( 16 * num_groups, viennacl::traits::context(in));
	viennacl::vector<unsigned int> global_offsets( 16 * num_groups, viennacl::traits::context(in));
	std::vector<unsigned int> cpu_global_offsets(16 * num_groups);
	int start_digit = 0;
	int end_digit = 0;

	struct
	{
		unsigned int start;
		unsigned int end;
	} scan_interval, scatter_interval;

	scan_interval.start = 0;
	scan_interval.end = in.size();

	viennacl::vector<uint64_t> *scan_vector, * scatter_vector, * temp;
	scan_vector = &in;
	scatter_vector = &out;
	int shift = 15;
	do {
		viennacl::ocl::enqueue( scan_kernel(*scan_vector,
											scan_interval.start,
											scan_interval.end,
											viennacl::ocl::local_mem(sizeof(cl_uint)  * 256),
											shift,
											global_histogram));
/*		std::cout << "global histogram" << std::endl;
		for (int i = 0 ; i <  num_groups; ++i)
		{
			std::cout << "group " << i << std::endl;
			for (int digit = 0 ; digit < 16 ; ++digit)
			{
				std::cout << "digit " << digit << " = " << global_histogram( digit * num_groups + i) << std::endl;
			}

		}
*/
		viennacl::linalg::exclusive_scan(global_histogram,global_offsets); // do on CPU ?
		viennacl::copy(global_offsets, cpu_global_offsets);
		start_digit=0;
		end_digit = 0;
		scatter_interval.start = scan_interval.start;
		scatter_interval.end = scan_interval.end;

		// skip on same digit.
		bool same_digit = true;
		for (end_digit = 1; end_digit < 16; ++end_digit)
		{
			int prev_digit_end =cpu_global_offsets[ end_digit * num_groups];
			if (prev_digit_end  && prev_digit_end < K)
				same_digit = false;

			if (!prev_digit_end)
				++start_digit;

			if (prev_digit_end <K)
				scan_interval.start = prev_digit_end;
			else
			{
				scan_interval.end = prev_digit_end;
				break;
			}
		}

		if (same_digit)
		{
			--shift;
			continue;
		}

		for (int digit = start_digit; digit < end_digit ; ++digit)
		{
			std::cout << digit << "=" << cpu_global_offsets[ digit * num_groups] << " ";
		}
		std::cout << std::endl;

		--end_digit;




		/*(__global uint* global_offset,
		int N,
		int first_digit,
		int last_digit,
		int shift,
		__global double* in,
		__global double * out,
			__global uint* keys_in,
			__global uint* keys_out,
			__local uint* local_positions)

		*/

		viennacl::ocl::enqueue( scatter_kernel(global_offsets,
				scatter_interval.start,
				scatter_interval.end,
				start_digit,
				end_digit,
				shift,
				*scan_vector,
				*scatter_vector,
				keys_in,
				keys_out,
				viennacl::ocl::local_mem(sizeof(cl_uint)  * 256 * 16)));
		shift--;
		temp = scatter_vector;
		scan_vector = scatter_vector;
		scatter_vector = temp;
		viennacl::copy(*scan_vector, trace);
		for (int i = 0 ;i < 1000 ; ++i)
			std::cout << trace[i] << " ";
		std::cout << std::endl;
	} while ((shift >=0) && (scatter_interval.end > K));

	if (scan_vector != &out)
		viennacl::copy(*scan_vector, out);

}



TEST(select_test, DISABLED_do_select)
{
	static viennacl::context g_context =   viennacl::ocl::current_context();
	static bool init = false;
	viennacl::ocl::context* ctx = g_context.opencl_pcontext();
	ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");

	std::vector<uint64_t> sample_data;
	for (int i = 0 ;i < 1000; i++)
		sample_data.push_back(i);

	int shift = 15;

	while (shift >= 0 )
	{
		int counts[16] = {};
		for (int i = 0 ;i < 1000; i++)
		{
			uint64_t val = sample_data[i];
			val = (val >> (4 * shift) ) & 0xF;
			counts[val]++;
		}
		std::cout << "shift " << shift << " ";
		for (int i = 0 ;i < 16 ; ++i)
			std::cout << counts[i] << " ";
		std::cout << std::endl;

		shift--;
	}
	viennacl::vector<uint64_t> sample_in(1000, g_context), sample_out(1000, g_context);
	viennacl::copy(sample_data, sample_in);
	sample_select(sample_in,99, sample_out);

}

