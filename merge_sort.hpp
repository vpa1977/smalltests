#ifndef MERGE_SORT_HPP
#define MERGE_SORT_HPP

#include <sstream>
#include <stdint.h>
#include "viennacl/context.hpp"
#include "viennacl/vector.hpp"


struct sort_value_types {
	template <typename T> 
	std::string type()  {
		return T::unimplemented;
	}
	template<>
	std::string type<uint32_t>()  { return "#define VALUE_TYPE uint\n"; }
	template<>
	std::string type<uint64_t>()  { return "#define VALUE_TYPE long\n"; }
	template<>
	std::string type<float>()  { return "#define VALUE_TYPE float\n"; }
	template<>
	std::string type<double>()  { return "#define VALUE_TYPE double\n"; }

};

template <typename basic_type> 
std::string generate_merge_sort_kernel()
{
	sort_value_types	 merge_sort_value_type;
	std::stringstream str;
	str << merge_sort_value_type.type<basic_type>() << std::endl
		<< "" << std::endl
		<< "uint lowerBoundBinarylocal(__local VALUE_TYPE* data, uint left, uint right, VALUE_TYPE searchVal){" << std::endl
		<< "   uint firstIndex = left;" << std::endl
		<< "   uint lastIndex = right;" << std::endl
		<< "   for (; firstIndex<lastIndex; ){" << std::endl
		<< "      uint midIndex = (firstIndex + lastIndex) / 2;" << std::endl
		<< "      double midValue = data[midIndex];" << std::endl
		<< "      " << std::endl
		<< "      if (midValue < searchVal){" << std::endl
		<< "         firstIndex = midIndex + 1;" << std::endl
		<< "      } else {" << std::endl
		<< "         lastIndex = midIndex;" << std::endl
		<< "      }" << std::endl
		<< "   }" << std::endl
		<< "   return(firstIndex);" << std::endl
		<< "}" << std::endl
		<< "uint upperBoundBinarylocal(__local VALUE_TYPE* data, uint left, uint right, VALUE_TYPE searchVal){" << std::endl
		<< "   uint upperBound = lowerBoundBinarylocal( data, left, right, searchVal);" << std::endl
		<< "   if (upperBound!=right){" << std::endl
		<< "      uint mid = 0;" << std::endl
		<< "      for (double upperValue = data[upperBound]; ( upperValue == searchVal) && upperBound<right; upperValue = data[upperBound]){" << std::endl
		<< "         mid = (upperBound + right) / 2;" << std::endl
		<< "         double midValue = data[mid];" << std::endl
		<< "         if (midValue == searchVal){" << std::endl
		<< "            upperBound = mid + 1;" << std::endl
		<< "         } else {" << std::endl
		<< "            right = mid;" << std::endl
		<< "            upperBound++;" << std::endl
		<< "         }" << std::endl
		<< "      }" << std::endl
		<< "   }" << std::endl
		<< "   return(upperBound);" << std::endl
		<< "}" << std::endl
		<< "__kernel void local_merge(" << std::endl
		<< "   uint vecSize, " << std::endl
		<< "   __global VALUE_TYPE *data_iter, " << std::endl
		<< "   __global uint* offsets," << std::endl
		<< "	__local VALUE_TYPE* lds," << std::endl
		<< "	__local VALUE_TYPE* lds2," << std::endl
		<< "	" << std::endl
		<< "	__local uint* indices," << std::endl
		<< "	__local uint* indices2" << std::endl
		<< "   " << std::endl
		<< "){" << std::endl

		<< "  { " << std::endl
		<< " 	size_t gloId = get_global_id( 0 ); " << std::endl
		<< " 	size_t groId = get_group_id( 0 ); " << std::endl
		<< " 	size_t locId = get_local_id( 0 ); " << std::endl
		<< " 	size_t wgSize = get_local_size( 0 ); " << std::endl
		<< " " << std::endl
		<< " " << std::endl
		<< " " << std::endl
		<< " 	double val; " << std::endl
		<< "	 if( gloId < vecSize) " << std::endl
		<< "	 { " << std::endl
		<< "	 	val = data_iter[ gloId ]; " << std::endl
		<< "	 	lds[locId ] = val; " << std::endl
		<< "	 	indices[locId] = gloId;" << std::endl
		<< "	 } " << std::endl
		<< "	 barrier( CLK_LOCAL_MEM_FENCE ); " << std::endl
		<< " " << std::endl
		<< " 	uint end = wgSize; " << std::endl
		<< " 	if( (groId+1)*(wgSize) >= vecSize ) " << std::endl
		<< " 		end = vecSize - (groId*wgSize); " << std::endl
		<< " " << std::endl
		<< " 	uint numMerges = 8; " << std::endl
		<< " 	uint pass; " << std::endl
		<< " 	for( pass = 1; pass <= numMerges; ++pass ) " << std::endl
		<< " 	{ " << std::endl
		<< " 		uint srcLogicalBlockSize = 1 << (pass-1); " << std::endl
		<< " 		if( gloId < vecSize) " << std::endl
		<< " 		{ " << std::endl
		<< " 			uint srcBlockNum = (locId) / srcLogicalBlockSize; " << std::endl
		<< " 			uint srcBlockIndex = (locId) % srcLogicalBlockSize; " << std::endl
		<< " " << std::endl
		<< " 			uint dstLogicalBlockSize = srcLogicalBlockSize<<1; " << std::endl
		<< " 			uint leftBlockIndex = (locId) & ~(dstLogicalBlockSize - 1 ); " << std::endl
		<< " " << std::endl
		<< " 			leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize; " << std::endl
		<< " 			leftBlockIndex = (((leftBlockIndex) < (end)) ? (leftBlockIndex) : (end)); " << std::endl
		<< " 			uint rightBlockIndex = (((leftBlockIndex + srcLogicalBlockSize) < (end)) ? (leftBlockIndex + srcLogicalBlockSize) : (end)); " << std::endl
		<< " " << std::endl
		<< "			uint insertionIndex = 0; " << std::endl
		<< " 			if(pass%2 != 0) " << std::endl
		<< " 			{ " << std::endl
		<< " 				if( (srcBlockNum & 0x1) == 0 ) " << std::endl
		<< " 				{ " << std::endl
		<< " 					insertionIndex = lowerBoundBinarylocal( lds, leftBlockIndex, rightBlockIndex, lds[ locId ]) - leftBlockIndex; " << std::endl
		<< " 				} " << std::endl
		<< " 				else " << std::endl
		<< " 				{ " << std::endl
		<< " 					insertionIndex = upperBoundBinarylocal( lds, leftBlockIndex, rightBlockIndex, lds[ locId ]) - leftBlockIndex; " << std::endl
		<< " 				} " << std::endl
		<< " 			} " << std::endl
		<< " 			else " << std::endl
		<< " 			{ " << std::endl
		<< " 				if( (srcBlockNum & 0x1) == 0 ) " << std::endl
		<< " 				{ " << std::endl
		<< " 					insertionIndex = lowerBoundBinarylocal( lds2, leftBlockIndex, rightBlockIndex, lds2[ locId ] ) - leftBlockIndex; " << std::endl
		<< " 				} " << std::endl
		<< " 				else " << std::endl
		<< " 				{ " << std::endl
		<< " 					insertionIndex = upperBoundBinarylocal( lds2, leftBlockIndex, rightBlockIndex, lds2[ locId ] ) - leftBlockIndex; " << std::endl
		<< " 				} " << std::endl
		<< " 			} " << std::endl
		<< " 			uint dstBlockIndex = srcBlockIndex + insertionIndex; " << std::endl
		<< " 			uint dstBlockNum = srcBlockNum/2;" << std::endl
		<< " 			uint offset =(dstBlockNum*dstLogicalBlockSize)+dstBlockIndex; " << std::endl
		<< " 			if(pass%2 != 0)" << std::endl
		<< " 			{ " << std::endl
		<< " 				 " << std::endl
		<< " 				lds2[offset ] = lds[ locId ];" << std::endl
		<< " 				indices2[offset] = indices[locId];" << std::endl
		<< " 			} " << std::endl
		<< " 			else " << std::endl
		<< " 			{" << std::endl
		<< " 				lds[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = lds2[ locId ];" << std::endl
		<< " 				indices[offset] = indices2[locId];" << std::endl
		<< " 			} " << std::endl
		<< " 		} " << std::endl
		<< " 		barrier( CLK_LOCAL_MEM_FENCE ); " << std::endl
		<< " 	} " << std::endl
		<< " 	if( gloId < vecSize) " << std::endl
		<< " 	{ " << std::endl
		<< " 		val = lds[ locId ]; " << std::endl
		<< " 		data_iter[ gloId ] = val;" << std::endl
		<< " 		offsets[gloId] = indices[locId]; " << std::endl
		<< " 	} " << std::endl
		<< " } " << std::endl
		<< "}" << std::endl
		<< "" << std::endl
		<< "" << std::endl
		<< "" << std::endl
		<< "uint lowerBoundBinary(__global VALUE_TYPE* source_iter, uint left, uint right, VALUE_TYPE searchVal){" << std::endl
		<< "   uint firstIndex = left;" << std::endl
		<< "   uint lastIndex = right;" << std::endl
		<< "   for (; firstIndex<lastIndex; ){" << std::endl
		<< "      uint midIndex = (firstIndex + lastIndex) / 2;" << std::endl
		<< "      double midValue = source_iter[midIndex];" << std::endl
		<< "      " << std::endl
		<< "      if (midValue < searchVal){" << std::endl
		<< "         firstIndex = midIndex + 1;" << std::endl
		<< "      } else {" << std::endl
		<< "         lastIndex = midIndex;" << std::endl
		<< "      }" << std::endl
		<< "   }" << std::endl
		<< "   return(firstIndex);" << std::endl
		<< "}" << std::endl
		<< "" << std::endl
		<< "uint upperBoundBinary( __global VALUE_TYPE* source_iter, uint left, uint right, VALUE_TYPE searchVal){" << std::endl
		<< "   uint upperBound = lowerBoundBinary( source_iter, left, right, searchVal);" << std::endl
		<< "   if (upperBound!=right){" << std::endl
		<< "      uint mid = 0;" << std::endl
		<< "      for (double upperValue = source_iter[upperBound]; searchVal ==  upperValue && upperBound<right; upperValue = source_iter[upperBound]){" << std::endl
		<< "         mid = (upperBound + right) / 2;" << std::endl
		<< "         double midValue = source_iter[mid];" << std::endl
		<< "         if (midValue ==  searchVal){" << std::endl
		<< "            upperBound = mid + 1;" << std::endl
		<< "         } else {" << std::endl
		<< "            right = mid;" << std::endl
		<< "            upperBound++;" << std::endl
		<< "         }" << std::endl
		<< "      }" << std::endl
		<< "   }" << std::endl
		<< "   return(upperBound);" << std::endl
		<< "}" << std::endl
		<< "__kernel void global_merge(" << std::endl
		<< "   uint srcVecSize, " << std::endl
		<< "   uint srcLogicalBlockSize, " << std::endl
		<< "   __global VALUE_TYPE *source_iter, " << std::endl
		<< "   __global VALUE_TYPE *result_iter, " << std::endl
		<< "   __global uint* src_offsets, " << std::endl
		<< "   __global uint* result_offsets" << std::endl
		<< ")" << std::endl
		<< "{" << std::endl
		<< "	size_t globalID = get_global_id( 0 ); " << std::endl
		<< " 	size_t groupID = get_group_id( 0 ); " << std::endl
		<< " 	size_t localID = get_local_id( 0 ); " << std::endl
		<< " 	size_t wgSize = get_local_size( 0 ); " << std::endl
		<< " " << std::endl
		<< "  " << std::endl
		<< " 	if( globalID >= srcVecSize ) " << std::endl
		<< " 		return; " << std::endl
		<< " 	uint srcBlockNum = globalID / srcLogicalBlockSize; " << std::endl
		<< " 	uint srcBlockIndex = globalID % srcLogicalBlockSize; " << std::endl
		<< " " << std::endl
		<< " " << std::endl
		<< " 	uint dstLogicalBlockSize = srcLogicalBlockSize<<1; " << std::endl
		<< " 	uint leftBlockIndex = globalID & ~(dstLogicalBlockSize - 1 ); " << std::endl
		<< " " << std::endl
		<< " 	leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize; " << std::endl
		<< " 	leftBlockIndex = (((leftBlockIndex) < (srcVecSize)) ? (leftBlockIndex) : (srcVecSize)); " << std::endl
		<< " 	uint rightBlockIndex = (((leftBlockIndex + srcLogicalBlockSize) < (srcVecSize)) ? (leftBlockIndex + srcLogicalBlockSize) : (srcVecSize)); " << std::endl
		<< " " << std::endl
		<< " 	uint insertionIndex = 0; " << std::endl
		<< " 	double search_val = source_iter[ globalID ]; " << std::endl
		<< " 	if( (srcBlockNum & 0x1) == 0 ) " << std::endl
		<< " 	{ " << std::endl
		<< " 		insertionIndex = lowerBoundBinary( source_iter, leftBlockIndex, rightBlockIndex, search_val ) - leftBlockIndex; " << std::endl
		<< " 	} " << std::endl
		<< " 	else " << std::endl
		<< " 	{ " << std::endl
		<< " 		insertionIndex = upperBoundBinary( source_iter, leftBlockIndex, rightBlockIndex, search_val ) - leftBlockIndex; " << std::endl
		<< " 	} " << std::endl
		<< " " << std::endl
		<< " 	uint dstBlockIndex = srcBlockIndex + insertionIndex; " << std::endl
		<< " 	uint dstBlockNum = srcBlockNum/2; " << std::endl
		<< " 	uint offset = (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex; " << std::endl
		<< " 	result_iter[ offset ] = source_iter[ globalID ]; " << std::endl
		<< " 	result_offsets[offset ] = src_offsets[ globalID];" << std::endl
		<< "" << std::endl
		<< "}" << std::endl

		<< "	__kernel void init_offsets(uint N, __global uint* in_offsets)" << std::endl
		<< "	{" << std::endl
		<< "		for (int id = get_global_id(0); id < N; id += get_global_size(0))" << std::endl
		<< "			in_offsets[id] = id;" << std::endl
		<< "	}" << std::endl

		<< std::endl;
	return str.str();
}

template <typename basic_type>
std::string generate_bitonic_sort_kernel()
{
	sort_value_types	 merge_sort_value_type;
	std::stringstream str;
	str << merge_sort_value_type.type<basic_type>() << std::endl
		<< "" << std::endl
		<< "/**********************************************************************" << std::endl
		<< "Copyright ©2014 Advanced Micro Devices, Inc. All rights reserved." << std::endl
		<< "" << std::endl
		<< "Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:" << std::endl
		<< "" << std::endl
		<< "•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer." << std::endl
		<< "•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or" << std::endl
		<< " other materials provided with the distribution." << std::endl
		<< "" << std::endl
		<< "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED" << std::endl
		<< " WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY" << std::endl
		<< " DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS" << std::endl
		<< " OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING" << std::endl
		<< " NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE." << std::endl
		<< "********************************************************************/" << std::endl
		<< "" << std::endl
		<< "/*" << std::endl
		<< " * For a description of the algorithm and the terms used, please see the" << std::endl
		<< " * documentation for this sample." << std::endl
		<< " *" << std::endl
		<< " * One invocation of this kernel, i.e one work thread writes two output values." << std::endl
		<< " * Since every pass of this algorithm does width/2 comparisons, each compare" << std::endl
		<< " * operation is done by one work thread." << std::endl
		<< " * " << std::endl
		<< " * Depending of the direction of sort for the work thread, the output values" << std::endl
		<< " * are written either as greater value to left element or lesser value to the" << std::endl
		<< " * left element. Right element and left element are the two elements we are " << std::endl
		<< " * comparing and \"left\" is the element with a smaller index into the array." << std::endl
		<< " *" << std::endl
		<< " * if direction is CL_TRUE, i.e evaluates to non zero, it means \"increasing\"." << std::endl
		<< " *" << std::endl
		<< " * For an explanation of the terms \"blockWidth\", \"sameDirectionBlockWidth\"," << std::endl
		<< " * stage, pass, pairDistance please go through the document shipped with this" << std::endl
		<< " * sample." << std::endl
		<< " *" << std::endl
		<< " * Since an explanation of the terms and the code here would be quite lengthy," << std::endl
		<< " * confusing and will greatly reduce the readability of this kernel, the code " << std::endl
		<< " * has been explained in detail in the document mentioned above." << std::endl
		<< " */" << std::endl
		<< "" << std::endl
		<< " " << std::endl
		<< "" << std::endl
		<< "__kernel " << std::endl
		<< "void bitonicSort(int N, __global VALUE_TYPE * theArray," << std::endl
		<< "				 __global uint* theOffsets," << std::endl
		<< "                 const uint stage, " << std::endl
		<< "                 const uint passOfStage," << std::endl
		<< "                 const uint direction)" << std::endl
		<< "{" << std::endl
		<< "    uint sortIncreasing = direction;" << std::endl
		<< "    uint threadId = get_global_id(0);" << std::endl
		<< "    " << std::endl
		<< "    uint pairDistance = 1 << (stage - passOfStage);" << std::endl
		<< "    uint blockWidth   = 2 * pairDistance;" << std::endl
		<< "" << std::endl
		<< "    uint leftId = (threadId % pairDistance) " << std::endl
		<< "                   + (threadId / pairDistance) * blockWidth;" << std::endl
		<< "" << std::endl
		<< "    uint rightId = leftId + pairDistance;" << std::endl
		<< "    " << std::endl
		<< "    VALUE_TYPE leftElement = theArray[leftId];" << std::endl
		<< "    VALUE_TYPE rightElement = theArray[rightId];" << std::endl
		<< "" << std::endl
		<< "	uint leftOffset = theOffsets[leftId];" << std::endl
		<< "	uint rightOffset = theOffsets[rightId];" << std::endl
		<< "    " << std::endl
		<< "    uint sameDirectionBlockWidth = 1 << stage;" << std::endl
		<< "    " << std::endl
		<< "    if((threadId/sameDirectionBlockWidth) % 2 == 1)" << std::endl
		<< "        sortIncreasing = 1 - sortIncreasing;" << std::endl
		<< "" << std::endl
		<< "    VALUE_TYPE greater;" << std::endl
		<< "    VALUE_TYPE lesser;" << std::endl
		<< "	uint lesserId; uint greaterId;" << std::endl
		<< "    if(leftElement > rightElement)" << std::endl
		<< "    {" << std::endl
		<< "		greaterId = leftOffset;" << std::endl
<< "		lesserId = rightOffset;" << std::endl
<< "        greater = leftElement;" << std::endl
<< "        lesser  = rightElement;" << std::endl
<< "    }" << std::endl
<< "    else" << std::endl
<< "    {" << std::endl
<< "		greaterId = rightOffset;" << std::endl
<< "		lesserId = leftOffset;" << std::endl
<< "" << std::endl
<< "        greater = rightElement;" << std::endl
<< "        lesser  = leftElement;" << std::endl
<< "    }" << std::endl
<< "    " << std::endl
<< "    if(sortIncreasing)" << std::endl
<< "    {" << std::endl
<< "        theArray[leftId]  = lesser;" << std::endl
<< "        theArray[rightId] = greater;" << std::endl
<< "" << std::endl
<< "        theOffsets[leftId]  = lesserId;" << std::endl
<< "        theOffsets[rightId] = greaterId;" << std::endl
<< "	}" << std::endl
<< "    else" << std::endl
<< "    {" << std::endl
<< "        theArray[leftId]  = greater;" << std::endl
<< "        theArray[rightId] = lesser;" << std::endl
<< "" << std::endl
<< "        theOffsets[leftId]  = greaterId;" << std::endl
<< "        theOffsets[rightId] = lesserId;" << std::endl
<< "" << std::endl
<< "    }" << std::endl
<< "}" << std::endl

<< "	__kernel void init_offsets(uint N, __global uint* in_offsets)" << std::endl
<< "	{" << std::endl
<< "		for (int id = get_global_id(0); id < N; id += get_global_size(0))" << std::endl
<< "			in_offsets[id] = id;" << std::endl
<< "	}" << std::endl

<< std::endl;
return str.str();
}

template<typename basic_type>
struct bitonic_sorter
{
	int length;
	viennacl::vector<unsigned int> src;
	viennacl::ocl::kernel kernel;
	viennacl::ocl::kernel init_offsets_kernel;

	bitonic_sorter(int size, const viennacl::context& ctx)
	{
		int v = size;
		// get next power of 2 
		v--;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++;
		length = v;
		src = viennacl::vector<unsigned int>(length, ctx);
		static bool init = false;
		viennacl::ocl::context& ocl_ctx = const_cast<viennacl::ocl::context&>(ctx.opencl_context());
		if (!init)
		{
			std::string program_text = generate_bitonic_sort_kernel<basic_type>();
			std::cout << "Device " << ocl_ctx.current_device().name() << std::endl;
			ocl_ctx.build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
			ocl_ctx.add_program(program_text, std::string("bitonic_sort"));
			init = true;
		}

		kernel = ocl_ctx.get_kernel("bitonic_sort", "bitonicSort");
		init_offsets_kernel = ocl_ctx.get_kernel("bitonic_sort", "init_offsets");
	}

	viennacl::vector<unsigned int> bitonic_sort(viennacl::vector<basic_type>& in)
	{

		int orig_size = in.size();
		viennacl::vector_range< viennacl::vector<basic_type> > fill_range(in, viennacl::range(in.size(), length));
		in.resize(length, true);
		viennacl::linalg::vector_assign(fill_range, std::numeric_limits<basic_type>::max());

		static int num_groups = 40;
		static int wg_size = 256;
		viennacl::context& the_context = viennacl::traits::context(in);
		viennacl::ocl::context& ctx = const_cast<viennacl::ocl::context&>(the_context.opencl_context());
		int size = src.size();
		viennacl::ocl::enqueue(init_offsets_kernel(size, src));

		cl_uint numStages = 0;
		cl_uint temp;

		cl_uint stage;
		cl_uint passOfStage;

		kernel.local_work_size(0, wg_size);
		kernel.global_work_size(0, std::max(length / 2, wg_size));

		init_offsets_kernel.local_work_size(0, wg_size);
		init_offsets_kernel.global_work_size(0, num_groups * wg_size);
		/*
		* This algorithm is run as NS stages. Each stage has NP passes.
		* so the total number of times the kernel call is enqueued is NS * NP.
		*
		* For every stage S, we have S + 1 passes.
		* eg: For stage S = 0, we have 1 pass.
		*     For stage S = 1, we have 2 passes.
		*
		* if length is 2^N, then the number of stages (numStages) is N.
		* Do keep in mind the fact that the algorithm only works for
		* arrays whose size is a power of 2.
		*
		* here, numStages is N.
		*
		* For an explanation of how the algorithm works, please go through
		* the documentation of this sample.
		*/

		/*
		* 2^numStages should be equal to length.
		* i.e the number of times you halve length to get 1 should be numStages
		*/
		for (temp = length; temp > 1; temp >>= 1)
		{
			++numStages;
		}


		for (stage = 0; stage < numStages; ++stage)
		{
			// Every stage has stage + 1 passes
			for (passOfStage = 0; passOfStage < stage + 1; ++passOfStage)
			{
				viennacl::ocl::enqueue(kernel((cl_uint)in.size(), in, src, stage, passOfStage, 1));
			}
		}
		return src;
	}


};



/** 
	sort provided vector and return offsets into original vector
*/
template<typename basic_type>
struct merge_sorter
{
	viennacl::vector<unsigned int> src;
	viennacl::vector<unsigned int> dst;
	viennacl::ocl::kernel local_kernel;
	viennacl::ocl::kernel global_kernel;
	viennacl::ocl::kernel init_offsets_kernel;
	int num_groups;

	merge_sorter(int size, const viennacl::context& ctx)
	{
		src = viennacl::vector<unsigned int>(size, ctx);
		dst = viennacl::vector<unsigned int>(size, ctx);
		static bool init = false;
		if (!init)
		{
			std::string program_text = generate_merge_sort_kernel<basic_type>();
			
			viennacl::ocl::context& ocl_ctx = const_cast<viennacl::ocl::context&>(ctx.opencl_context());
			std::cout << "Device " << ocl_ctx.current_device().name() << std::endl;
			ocl_ctx.build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
			ocl_ctx.add_program(program_text, std::string("merge_sort"));
			init = true;
		}
		local_kernel = viennacl::ocl::current_context().get_kernel("merge_sort", "local_merge");
		global_kernel = viennacl::ocl::current_context().get_kernel("merge_sort", "global_merge");
		init_offsets_kernel = viennacl::ocl::current_context().get_kernel("merge_sort", "init_offsets");
		num_groups = ctx.opencl_context().current_device().max_compute_units() * 4 + 1;
	}


	viennacl::vector<unsigned int> merge_sort(viennacl::vector<basic_type>& in)
	{
		
		int wg_size = 128;
		

		size_t localRange = wg_size;
		size_t globalRange = in.size();
		size_t modlocalRange = globalRange & (localRange - 1);
		if (modlocalRange > 0)
		{
			globalRange &= ~modlocalRange;
			globalRange += localRange;
		}

		local_kernel.local_work_size(0, wg_size);
		global_kernel.local_work_size(0, wg_size);
		init_offsets_kernel.local_work_size(0, wg_size);
		init_offsets_kernel.global_work_size(0, wg_size* num_groups);

		local_kernel.global_work_size(0, globalRange);
		global_kernel.global_work_size(0, globalRange);


		cl_uint size = src.size();
		viennacl::ocl::enqueue(init_offsets_kernel(size, src));

		viennacl::ocl::enqueue(local_kernel((cl_uint)in.size(), in, src, viennacl::ocl::local_mem(wg_size * sizeof(cl_double)), viennacl::ocl::local_mem(wg_size * sizeof(cl_double)),
			viennacl::ocl::local_mem(wg_size * sizeof(cl_uint)), viennacl::ocl::local_mem(wg_size * sizeof(cl_uint))));
		if (in.size() <= localRange)
			return src;



		viennacl::vector<basic_type> tmp(in.size(), viennacl::traits::context(in));

		// An odd number of elements requires an extra merge pass to sort
		int numMerges = 0;
		// Calculate the log2 of vecSize, taking into account our block size
		// from kernel 1 is 256
		// this is how many merge passes we want
		int log2BlockSize = in.size() >> 8;

		for (; log2BlockSize > 1; log2BlockSize >>= 1) {
			++numMerges;
		}
		// Check to see if the input vector size is a power of 2, if not we will
		// need last merge pass
		int vecPow2 = (in.size() & (in.size() - 1));
		numMerges += in.size() > 0 ? 1 : 0;

		for (int pass = 1; pass <= numMerges; ++pass) {
			int srcLogicalBlockSize = localRange << (pass - 1);


			if ((pass & 0x1) > 0) {
				viennacl::ocl::enqueue(global_kernel((cl_uint)in.size(), (cl_uint)srcLogicalBlockSize, in, tmp, src, dst));
			}
			else {
				viennacl::ocl::enqueue(global_kernel((cl_uint)in.size(), (cl_uint)srcLogicalBlockSize, tmp, in, dst, src));
			}

		}

		if ((numMerges & 1) > 0)
		{
			tmp.fast_swap(in);
			dst.fast_swap(src);
		}
		return src;
	}


};








#endif