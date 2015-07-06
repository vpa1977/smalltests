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


TEST(reduce, DISABLED_reduce_func)
{
	viennacl::ocl::current_context().cache_path("c:/tmp/");
	static viennacl::context g_context = viennacl::ocl::current_context();
	static bool init = false;
	viennacl::ocl::context* ctx = g_context.opencl_pcontext();
	ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
	std::cout << "Device " << ctx->current_device().name() << std::endl;
	{
		int size = 160000;
		viennacl::vector<double> test(size, g_context);
		std::vector<double> cpu(size);
		for (int i = 0; i < cpu.size(); ++i)
			cpu[i] = 1;
		viennacl::copy(cpu, test);
		double res = viennacl::ml::opencl::reduce(test);
		ASSERT_EQ(res, cpu.size());

	}

}