#ifndef BUCKET_SELECT_HPP
#define BUCKET_SELECT_HPP
#include "viennacl/vector.hpp"
/*
Select N smallest values from basic_type vector
*/

#include "viennacl/context.hpp"
#include "viennacl/vector.hpp"
/*
Select N smallest values from basic_type vector
*/
template<typename basic_type>
viennacl::vector<unsigned int> bucket_select(int N, const viennacl::vector<basic_type>& in)
{
	viennacl::vector<unsigned int> src(in.size(), viennacl::traits::context(in));
	viennacl::vector<unsigned int> dst(in.size() , viennacl::traits::context(in));

	// load kernels
	static bool init = false;
	static int num_groups = 1;
	static int wg_size = 128;
	

	if (!init)
	{
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
		init = true;
	}


	viennacl::ocl::kernel scan_kernel = viennacl::ocl::current_context().get_kernel("test", "scan_buckets");
	viennacl::ocl::kernel scatter_kernel = viennacl::ocl::current_context().get_kernel("test", "scatter_buckets");
	viennacl::ocl::kernel init_offsets_kernel = viennacl::ocl::current_context().get_kernel("test", "init_offsets");

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

	int num_buckets = 10;
	viennacl::vector<unsigned int> global_histogram((num_buckets + 1) * num_groups, viennacl::traits::context(in)); // -wg_size
	viennacl::vector<unsigned int> global_histogram_prefix((num_buckets + 1) * num_groups  + 1, viennacl::traits::context(in));
	std::vector< unsigned int > global_histogram_cpu((num_buckets + 1) * num_groups + 1);
	int scan_start = 0;
	int scan_end = in.size();
	basic_type pivot;
	basic_type base_value;
	int split_bucket = 0;
	base_value = 0;
	pivot = std::numeric_limits<basic_type>::max() / num_buckets;
	assert(pivot > 0);
	while (position < N)
	{
		int main = (scan_end / wg_size) * wg_size; // floor to multiple wg size
		int loop_end = main == scan_end ? main : main + wg_size; // add wg size if needed

		viennacl::ocl::enqueue(scan_kernel(in,
			src,
			scan_end,
			loop_end,
			viennacl::ocl::local_mem(sizeof(cl_uint) *wg_size),
			base_value,
			pivot,
			num_buckets,
			global_histogram));

		viennacl::linalg::exclusive_scan(global_histogram, global_histogram_prefix);
		viennacl::copy(global_histogram_prefix, global_histogram_cpu);
		global_histogram_cpu[global_histogram_cpu.size() - 1] = global_histogram_cpu[global_histogram_cpu.size() - 2]; // fix last element

		for (split_bucket = 1; split_bucket < num_buckets; ++split_bucket)
		{
			int offset = global_histogram_cpu[num_groups * split_bucket];
			if (offset >= N)
				break;
		}
		viennacl::ocl::enqueue(scatter_kernel(
			in,
			src,
			scan_end,
			loop_end,
			viennacl::ocl::local_mem(sizeof(cl_uint) *wg_size),
			(basic_type)base_value,
			(basic_type)pivot,
			num_buckets,
			split_bucket,
			global_histogram_prefix,
			dst
			));

		int hist_max = global_histogram_cpu[num_groups * split_bucket];
		int hist_min = global_histogram_cpu[num_groups * (split_bucket - 1)];
		//#ifdef DEBUG_RADIX_SELECT
		std::vector<unsigned int> dst_cpu(in.size());
		std::vector<unsigned int> src_cpu(in.size());
		viennacl::copy(dst, dst_cpu);
		viennacl::copy(src, src_cpu);
		//#endif

		if (hist_max == N)
			break;
		if (hist_max> N && hist_min < N)
		{
			scan_start = global_histogram_cpu[num_groups * (split_bucket - 1)];
			scan_end = global_histogram_cpu[num_groups * split_bucket];
			if (scan_start > 0)
			{
				viennacl::copy(dst.begin(), dst.begin() + scan_start, result.begin() + position);
				position += scan_start;
			}
			//#ifdef DEBUG_RADIX_SELECT
			std::vector<unsigned int> result_cpu(in.size());
			viennacl::copy(result, result_cpu);
			//#endif
			if (position >= N)
				break;
			if (scan_end == dst.size() && scan_start == 0)
				dst.fast_swap(src);
			else
				viennacl::copy(dst.begin() + scan_start, dst.begin() + scan_end, src.begin());
			scan_end -= scan_start;
		}

		base_value += pivot * (split_bucket-1);
		// update pivot
		
		pivot = pivot / num_buckets;
		if (pivot == 0)
			break;


	}
	if (position <N)
		viennacl::copy(dst.begin(), dst.begin() + (N - position), result.begin() + position);

	return result;
}






#endif