/*
 * distance.cpp
 *
 *  Created on: 11/06/2015
 *      Author: bsp
 */


#include "gtest/gtest.h"

#include "viennacl/context.hpp"
#include "viennacl/ml/knn.hpp"

#include <boost/numeric/ublas/matrix.hpp>
#include <atomic>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>


using namespace viennacl::ml::knn;
using namespace std::chrono;

dense_sliding_window prepare_window(viennacl::context& ctx, int num_attributes, int num_instances)
{
	using namespace boost::numeric::ublas;
	matrix<double> cpu( num_instances, num_attributes);
	dense_sliding_window window(ctx, num_attributes, num_instances);
	for (int row = 0; row < num_instances; ++row )
	{
		for (int column = 0; column < num_attributes; ++column)
			cpu(row, column) = row;
	}
	viennacl::copy( cpu , window.m_values_window);
	return window;
}

void split_calc_distance(std::vector<double>& to_sort,viennacl::ocl::context* p_context, int num_splits, naive_knn& knn, viennacl::vector<double>& distances, dense_sliding_window& sliding_window, int num_instances, viennacl::vector<double>& sample)
{
	int len = num_instances / num_splits;
	auto gpu_begin = distances.begin();
	auto gpu_end = gpu_begin + len;

	int last = num_instances - len * num_splits;
	int current = 0;
	knn.calc_distance(distances, sliding_window, current, current+len, sample);
	current += len;
	for (; current < num_instances; current += len)
	{
		p_context->get_queue().finish();
		viennacl::copy(gpu_begin, gpu_end, to_sort.begin());
		knn.calc_distance(distances, sliding_window, current, current+len, sample);
		std::sort(to_sort.begin(), to_sort.end());
	}
	p_context->get_queue().finish();
	viennacl::copy(gpu_begin, gpu_end, to_sort.begin());
	std::sort(to_sort.begin(), to_sort.end());
	if (last > 0)
	{
		//knn.calc_distance(distances, sliding_window, current -len, current + last, sample);

	}

}


void compare_dist_calc(int num_attributes, int num_instances)
{
	std::cout << "Test:" << num_attributes << " - " << num_instances << std::endl ;
	static viennacl::context g_context =   viennacl::ocl::current_context();
	static bool init = false;
	if (!init)
	{
		viennacl::ocl::context* ctx = g_context.opencl_pcontext();
		ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
	}
	static viennacl::context m_context(viennacl::MAIN_MEMORY);
	dense_sliding_window g_test = prepare_window(g_context, num_attributes, num_instances);
	//dense_sliding_window m_test = prepare_window(m_context, num_attributes, num_instances);
	viennacl::vector<double> g_sample(num_attributes,g_context);
//	viennacl::vector<double> m_sample(num_attributes,m_context);

	viennacl::vector<double> g_distances( num_instances, g_context);
//	viennacl::vector<double> m_distances( num_instances, m_context);

	naive_knn g_naive_knn(g_context);
	//naive_knn m_naive_knn(m_context);
	g_naive_knn.update_bounds(g_test);
////	m_naive_knn.update_bounds(m_test);

	g_naive_knn.calc_distance(g_distances, g_test, 0, num_instances, g_sample);
//	m_naive_knn.calc_distance(m_distances, m_test, 0, num_instances, m_sample);
	std::cout << "Ready:" << std::endl ;
	{
		std::vector<double> to_sort(g_distances.size());
		for (int splits = 1 ; splits <=16; splits = splits *2)
		{
			system_clock::time_point t1 = system_clock::now();
			for (int i = 0 ; i < 100 ; i ++ )
			{

				g_naive_knn.calc_distance(g_distances, g_test, 0, num_instances/splits, g_sample);
				g_context.opencl_context().get_queue().finish();
				viennacl::copy(g_distances, to_sort);
				std::sort(to_sort.begin(), to_sort.begin()+(num_instances/splits));
				for (int i = 0 ;i < to_sort.size() ; ++i)
				{
					double value = i * i * num_attributes;
					assert( to_sort[i] == value);
				}
			}
			system_clock::time_point t2 = system_clock::now();

			duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
			std::cout << "Kernel count " << (num_instances/splits) << " GPU:" << time_span.count() << "," ;
		}
	}

	{
		steady_clock::time_point t1 = steady_clock::now();
		for (int i = 0 ; i < 100 ; i ++ )
		{
			int num_splits = 1;
			std::vector<double> to_sort(num_instances/num_splits);
			split_calc_distance(to_sort,g_context.opencl_pcontext(), num_splits, g_naive_knn, g_distances, g_test, num_instances, g_sample);
		}
		steady_clock::time_point t2 = steady_clock::now();

		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		std::cout << "CPU:" << time_span.count() << std::endl ;
	}

}


class KnnVariousTests : public ::testing::Test
{
public:
	viennacl::context g_context;
	viennacl::vector<double> result;
	viennacl::vector<double> fetch;
	viennacl::vector<double> fetch1;
	KnnVariousTests()
	{
	}

	void SetUp()
	{
		std::string code;
		const char* const distance_simple =
				"\n__kernel void knn_calc_distance(int start_row, int end_row,__global double* fetch,  __global double* result)"
				"{\n"
				" 	for (int id = get_global_id(0) + start_row;id < end_row; id += get_global_size(0) )"
				" 	{"
				"       double cur = 0;"
				"		result[id-start_row] = fetch[id-start_row]; "
				" 	} "
				"}\n" ;
		code.append(distance_simple);

		const char* const distance_simple_double =
				"\n__kernel void knn_calc_distance1(int start_row, int end_row,__global double* fetch1,__global double* fetch,  __global double* result)"
				"{\n"
				" 	for (int id = get_global_id(0) + start_row;id < end_row; id += get_global_size(0) )"
				" 	{"
				"       double cur = 0;"
				"		result[id-start_row] = fetch[id-start_row] + fetch1[id-start_row]; "
				" 	} "
				"}\n" ;
		code.append(distance_simple_double);

		g_context =   viennacl::ocl::current_context();
		viennacl::ocl::context* ctx = g_context.opencl_pcontext();
		ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
		ctx->add_program(code, "test_speed" );
		fetch = viennacl::vector<double>(1024*1024, g_context);
		fetch1 = viennacl::vector<double>(1024*1024, g_context);
		result = viennacl::vector<double>(1024*1024, g_context);

	}
};

TEST_F(KnnVariousTests, DISABLED_ignore)
{
}

TEST_F(KnnVariousTests, DISABLED_bestGlobalSize)
{
	int global_time = 0;
	int global_size = 256;
	int num_cores = 1;
	int max_cores = 64;
	for (num_cores = 1 ; num_cores < max_cores ; ++num_cores)
	{
		double cnt = 0;
		for (int iter = 0 ; iter < 30 ; ++iter )
		{
			steady_clock::time_point t1 = steady_clock::now();

			viennacl::ocl::context* ctx = g_context.opencl_pcontext();
			for (int i = 0; i < 10000 ; ++i)
			{
				viennacl::ocl::kernel& dist = ctx->get_kernel("test_speed", "knn_calc_distance");
				dist.local_work_size(0,256);
				dist.global_work_size(0, num_cores*global_size);
				viennacl::ocl::enqueue(dist(0,max_cores*global_size ,fetch,result));
			}
			ctx->get_queue().finish();

			steady_clock::time_point t2 = steady_clock::now();

			duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
			cnt += time_span.count();
		}
		cnt = cnt/30;
		std::cout << "NUM cores " << num_cores << " GlobalSize :" << (global_size * num_cores) << " time " << cnt << std::endl ;
	}
}


TEST_F(KnnVariousTests, DISABLED_bestGlobalSize_2)
{
	int global_time = 0;
	int global_size = 256;
	int num_cores = 1;
	int max_cores = 64;
	for (num_cores = 1 ; num_cores < max_cores ; ++num_cores)
	{
		double cnt = 0;
		for (int iter = 0 ; iter < 30 ; ++iter )
		{
			steady_clock::time_point t1 = steady_clock::now();

			viennacl::ocl::context* ctx = g_context.opencl_pcontext();
			for (int i = 0; i < 10000 ; ++i)
			{
				viennacl::ocl::kernel& dist = ctx->get_kernel("test_speed", "knn_calc_distance1");
				dist.local_work_size(0,256);
				dist.global_work_size(0, num_cores*global_size);
				viennacl::ocl::enqueue(dist(0,max_cores*global_size ,fetch1,fetch,result));
			}
			ctx->get_queue().finish();

			steady_clock::time_point t2 = steady_clock::now();

			duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
			cnt += time_span.count();
		}
		cnt = cnt/30;
		std::cout << "NUM cores " << num_cores << " GlobalSize :" << (global_size * num_cores) << " time " << cnt << std::endl ;
	}

}



