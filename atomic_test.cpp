#include <vector>
#include "gtest/gtest.h"
#include "viennacl/context.hpp"
#include "viennacl/ml/knn.hpp"
#include "viennacl/vector.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <atomic>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>


TEST(Atomics, DISABLED_test_lock)
{
	viennacl::ocl::current_context().cache_path("c:/tmp/");
	viennacl::ocl::context& g_context = viennacl::ocl::current_context();
	

	std::for_each(g_context.devices().begin(), g_context.devices().end(), [] (const viennacl::ocl::device& device){
		std::cout << device.id() << " " << device.name() << std::endl;
	});
	//viennacl::ocl::current_context().switch_device(0);
	std::cout << "Now using " << g_context.current_device().name() << std::endl;

	static bool init = false;
	viennacl::ocl::context* ctx = &g_context;
	ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");

	FILE * tmp = fopen("atomic_test.cl", "rb");
	fseek(tmp, 0, SEEK_END);
	std::vector<char> binary;
	binary.resize(ftell(tmp));
	rewind(tmp);
	fread(&binary[0], binary.size(), 1, tmp);
	fclose(tmp);
	binary.push_back(0);
	std::string program_text(&binary[0]);
	ctx->add_program(program_text, std::string("test"));

	viennacl::ocl::kernel kernel = ctx->get_kernel("test", "test_atomic_lock");
	kernel.local_work_size(0, 16);
	kernel.global_work_size(0, 16);

	viennacl::vector<double> output(10, g_context);
	viennacl::vector<unsigned int> locks(kernel.global_work_size(0), g_context);
	viennacl::linalg::vector_assign(locks, (unsigned int)0);

	viennacl::ocl::enqueue(kernel(output));
	ctx->current_queue().finish();
	ASSERT_EQ(output(0), 16.0);
	ASSERT_TRUE(1);
}