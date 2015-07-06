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


TEST(bitonic, DISABLED_compile)
{
	viennacl::ocl::current_context().cache_path("c:/tmp/");
	FILE * tmp = fopen("bitonic_sort.cl", "rb");
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

	viennacl::ocl::kernel bsort_merge = ctx->get_kernel("test", "bsort_merge");
	viennacl::ocl::kernel bsort_merge_last = ctx->get_kernel("test", "bsort_merge_last");


}