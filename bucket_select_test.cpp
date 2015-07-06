#include <vector>
#include "gtest/gtest.h"
#include "radix_select.hpp"
#include "bucket_select.hpp"
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
template <typename basictype>
 void print_vector(const viennacl::vector<basictype>&v)
{
	for (auto val : v)
	{
		std::cout << " " << val;
	}
	std::cout << std::endl;
}

static void print_vector_selected(const viennacl::vector<float>&v, const viennacl::vector<unsigned int>& selected)
{
	std::vector<unsigned int> test;
	for (auto idx : selected)
	{
		test.push_back(v(idx));
	}
	std::stable_sort(test.begin(), test.end());
	for (auto val : test)
		std::cout << " " << val;
	std::cout << std::endl;
}

// 1. add printf for select
// 2. 
TEST(bucket_sort, DISABLED_real_sort)
{
	viennacl::ocl::current_context().cache_path("c:/tmp/");
	viennacl::vector<float> selection(1000000, viennacl::ocl::current_context());
	std::vector<float> test(1000000);

	for (int i = 0; i < 1000000; i++)
		test[i] = i;
	
	viennacl::copy(test, selection);
	viennacl::vector<unsigned int> offsets = bucket_select(318, selection);
	std::vector<unsigned int> cpu_off(offsets.size());
	viennacl::copy(offsets, cpu_off);
	print_vector_selected(selection, offsets);
	ASSERT_EQ(offsets.size(), 318);
}


TEST(bucket_sort, DISABLED_test_bucket_assign)
{
	viennacl::ocl::current_context().cache_path("c:/tmp/");
	FILE * tmp = fopen("bucket_select.cl", "rb");
	fseek(tmp, 0, SEEK_END);
	std::vector<char> binary;
	binary.resize(ftell(tmp));
	rewind(tmp);
	fread(&binary[0], binary.size(), 1, tmp);
	fclose(tmp);
	binary.push_back(0);
	static viennacl::context g_context = viennacl::ocl::current_context();
	static bool init = false;
	viennacl::ocl::context* ctx = g_context.opencl_pcontext();
	std::cout << "Device " << ctx->current_device().name() << std::endl;
	ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
	std::string program_text(&binary[0]);
	ctx->add_program(program_text, std::string("test"));

	viennacl::ocl::kernel assign_buckets_kernel =  ctx->get_kernel("test", "assign_buckets");
	viennacl::ocl::kernel init_offsets_kernel = ctx->get_kernel("test", "init_offsets");
	int N = 10;
	int max = 10;
	int num_buckets = 10;
	std::vector<int> cpu_in;

	for (int i = 0; i < 10; i++)
		cpu_in.push_back(i);

	viennacl::vector<int> in(N, *ctx);
	viennacl::vector<unsigned int> src(N, *ctx);
	viennacl::vector<unsigned int> dst(N, *ctx);
	std::vector<unsigned int> cpu_dst(N);
	
	viennacl::copy(cpu_in, in);
	int wg_size = 128;
	int scan_end = in.size();
	int main = (scan_end / wg_size) * wg_size; // floor to multiple wg size
	int loop_end = main == scan_end ? main : main + wg_size; // add wg size if needed
	int pivot = max / num_buckets;

	viennacl::ocl::enqueue(init_offsets_kernel(scan_end, src));
	viennacl::ocl::enqueue(assign_buckets_kernel(in, src, scan_end, loop_end, 0, 1, num_buckets, dst));
	
	print_vector(dst);
	printf("");
}