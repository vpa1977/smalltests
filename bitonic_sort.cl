#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/* Sort elements within a vector */
#define VECTOR_SORT(input, dir, indices)                          \
   comp = input < shuffle(input, as_ulong2(mask2)) ^ dir;                    \
   input = shuffle(input, as_ulong2(comp * 2 + add2));             \
   indices = shuffle( indices, as_ulong2(comp*2 + add2));          \
   comp = input < shuffle(input, as_ulong2(mask1)) ^ dir;                    \
   input = shuffle(input, as_ulong2(comp + add1));                 \
   indices = shuffle(indices, as_ulong2(comp + add1));             \
   

#define VECTOR_SWAP(input1, input2, dir, indices1,indices2)       \
   temp = input1;                                                 \
   temp_indices = indices1;                                       \
   comp = (input1 < input2 ^ dir) * 4 + add3;                     \
   input1 = shuffle2(input1, input2, as_ulong2(comp));             \
   input2 = shuffle2(input2, temp, as_ulong2(comp));               \
   indices1 = shuffle2( indices1, indices2, as_ulong2(comp));      \
   indices2 = shuffle2( indices2, temp_indices, as_ulong2(comp));  \
   

/* Perform initial sort */
__kernel void bsort_init(__global double4 *g_data,  __local double4 *l_data, __global int4* g_indices, __local int4* l_indices) {

   int dir;
   uint id, global_start, size, stride;
   double4 input1, input2, temp;
   int4 indices1, indices2, temp_indices;
   
   int4 comp;

   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(1, 2, 2, 3);
   
   int gid = get_global_id(0) * 8;

   id = get_local_id(0) * 2;
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   input1 = g_data[global_start]; 
   input2 = g_data[global_start+1];
   
   //indices1 = g_indices[ global_start ];
   //indices2 = g_indices[ global_start +1];
   
   indices1 = (int4)( gid, gid + 1, gid + 2 , gid +3); 
   indices2 = (int4)( gid+4, gid + 5, gid +6, gid +7);
   
   //printf(" global_di %d, indice %d, %d\n", get_global_id(0), global_start);
   

   /* Sort input 1 - ascending */
   comp = input1 < shuffle(input1, as_ulong2(mask1));
   
   input1 = shuffle(input1, as_ulong2(comp + add1));
   indices1 = shuffle( indices1, as_ulong2( comp + add1));
   
   comp = input1 < shuffle(input1, as_ulong2(mask2));
   input1 = shuffle(input1, as_ulong2(comp * 2 + add2));
   indices1 = shuffle(indices1, as_ulong2(comp*2 + add2));
   
   comp = input1 < shuffle(input1, as_ulong2(mask3));
   input1 = shuffle(input1, as_ulong2(comp + add3));
   indices1 = shuffle(indices1, as_ulong2( comp + add3));

   /* Sort input 2 - descending */
   comp = input2 > shuffle(input2, mask1);
   input2 = shuffle(input2, as_uint4(comp + add1));
   indices2 = shuffle(indices2, as_uint4(comp + add1));
   
   comp = input2 > shuffle(input2, mask2);
   input2 = shuffle(input2, as_uint4(comp * 2 + add2));
   indices2 = shuffle(indices2, as_uint4(comp * 2 + add2));
   
   comp = input2 > shuffle(input2, mask3);
   input2 = shuffle(input2, as_uint4(comp + add3));     
   indices2 = shuffle(indices2, as_uint4(comp + add3));

   /* Swap corresponding elements of input 1 and 2 */
   add3 = (int4)(4, 5, 6, 7);
   dir = get_local_id(0) % 2 * -1;
   temp = input1;
   temp_indices = indices1;
   
   comp = (input1 < input2 ^ dir) * 4 + add3;
   
   input1 = shuffle2(input1, input2, as_uint4(comp));
   indices1 = shuffle2(indices1, indices2, as_uint4(comp));
   
   input2 = shuffle2(input2, temp, as_uint4(comp));
   indices2 = shuffle2(indices2, temp_indices, as_uint4(comp));

   /* Sort data and store in local memory */
   VECTOR_SORT(input1, dir, indices1);
   VECTOR_SORT(input2, dir, indices2);
   
   l_data[id] = input1;
   l_indices[id] = indices1;
   
   l_data[id+1] = input2;
   l_indices[id+1] = indices2;

   /* Create bitonic set */
   for(size = 2; size < get_local_size(0); size <<= 1) {
      dir = (get_local_id(0)/size & 1) * -1;

      for(stride = size; stride > 1; stride >>= 1) {
         barrier(CLK_LOCAL_MEM_FENCE);
         id = get_local_id(0) + (get_local_id(0)/stride)*stride;
         VECTOR_SWAP(l_data[id], l_data[id + stride], dir, l_indices[id], l_indices[id+stride])
      }

      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) * 2;
      
      input1 = l_data[id];
      indices1 = l_indices[id];
       
      input2 = l_data[id+1];
      indices2 = l_indices[id+1];
      
      temp = input1;
      temp_indices = indices1;
      
      comp = (input1 < input2 ^ dir) * 4 + add3;
      
      input1 = shuffle2(input1, input2, as_uint4(comp));
      indices1 = shuffle2(indices1, indices2, as_uint4(comp));
      
      input2 = shuffle2(input2, temp, as_uint4(comp));
      indices2 = shuffle2(indices2, temp_indices, as_uint4(comp));
      
      VECTOR_SORT(input1, dir, indices1);
      VECTOR_SORT(input2, dir, indices2);
      
      l_data[id] = input1;
   	  l_indices[id] = indices1;
   
   	  l_data[id+1] = input2;
   	  l_indices[id+1] = indices2;
   }

   /* Perform bitonic merge */
   dir = (get_group_id(0) % 2) * -1;
   for(stride = get_local_size(0); stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data[id], l_data[id + stride], dir, l_indices[id], l_indices[id+stride])
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   
      input1 = l_data[id];
      indices1 = l_indices[id];
       
      input2 = l_data[id+1];
      indices2 = l_indices[id+1];
      
      temp = input1;
      temp_indices = indices1;
      
      comp = (input1 < input2 ^ dir) * 4 + add3;
      
      input1 = shuffle2(input1, input2, as_uint4(comp));
      indices1 = shuffle2(indices1, indices2, as_uint4(comp));
      
      input2 = shuffle2(input2, temp, as_uint4(comp));
      indices2 = shuffle2(indices2, temp_indices, as_uint4(comp));
      
      VECTOR_SORT(input1, dir, indices1);
      VECTOR_SORT(input2, dir, indices2);
      
   g_indices[global_start] = indices1;
   g_indices[global_start+1] = indices2;
         
   g_data[global_start] = input1;
   g_data[global_start+1] = input2;
}

/* Perform lowest stage of the bitonic sort */
__kernel void bsort_stage_0(__global double4 *g_data, __local double4 *l_data, 
                            uint high_stage, __global int4* g_indices, __local int4* l_indices) {

   int dir;
   uint id, global_start, stride;
   double4 input1, input2, temp;
   int4 indices1, indices2, temp_indices;
   int4 comp;

   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(4, 5, 6, 7);

   /* Determine data location in global memory */
   id = get_local_id(0);
   dir = (get_group_id(0)/high_stage & 1) * -1;
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   /* Perform initial swap */
   input1 = g_data[global_start];
   input2 = g_data[global_start + get_local_size(0)];
   
   indices1 = g_indices[global_start];
   indices2 = g_indices[global_start + get_local_size(0)];
   
   comp = (input1 < input2 ^ dir) * 4 + add3;
   
   l_data[id] = shuffle2(input1, input2, as_uint4(comp));
   l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));
   
   l_indices[id] = shuffle2(indices1, indices2, as_uint4(comp));
   l_indices[id + get_local_size(0)] = shuffle2(indices2, indices1, as_uint4(comp));
   

   /* Perform bitonic merge */
   for(stride = get_local_size(0)/2; stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data[id], l_data[id + stride], dir, l_indices[id], l_indices[id+stride])
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   
   input1 = l_data[id]; 
   input2 = l_data[id+1];
   
   indices1 = l_indices[id];
   indices2 = l_indices[id+1];
      
   temp = input1;
   temp_indices = indices1;
   
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));
   
   indices1 = shuffle2(indices1, indices2, as_uint4(comp));
   indices2 = shuffle2(indices2, temp_indices, as_uint4(comp));
   VECTOR_SORT(input1, dir, indices1);
   VECTOR_SORT(input2, dir, indices2);

   /* Store output in global memory */
   g_indices[global_start + get_local_id(0)] = indices1;
   g_indices[global_start + get_local_id(0) + 1] = indices2;
   
   g_data[global_start + get_local_id(0)] = input1;
   g_data[global_start + get_local_id(0) + 1] = input2;
}

/* Perform successive stages of the bitonic sort */
__kernel void bsort_stage_n(__global double4 *g_data, __local double4 *l_data, 
                            uint stage, uint high_stage, __global int4* g_indices, __local int4* l_indices) {

   int dir;
   double4 input1, input2;
   int4 indices1, indices2;
   int4 comp, add;
   uint global_start, global_offset;

   add = (int4)(4, 5, 6, 7);

   /* Determine location of data in global memory */
   dir = (get_group_id(0)/high_stage & 1) * -1;
   global_start = (get_group_id(0) + (get_group_id(0)/stage)*stage) *
                   get_local_size(0) + get_local_id(0);
   global_offset = stage * get_local_size(0);

   /* Perform swap */
   input1 = g_data[global_start];
   input2 = g_data[global_start + global_offset];
   
   indices1 = g_indices[global_start];
   indices2 = g_indices[global_start + global_offset];
   
   comp = (input1 < input2 ^ dir) * 4 + add;
   
   g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
   g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
   
   g_indices[global_start] = shuffle2(indices1, indices2, as_uint4(comp));
   g_indices[global_start + global_offset] = shuffle2(indices2, indices1, as_uint4(comp));
}

/* Sort the bitonic set */
__kernel void bsort_merge(__global double4 *g_data, __local double4 *l_data, uint stage, int dir, __global int4* g_indices, __local int4* l_indices) {

   double4 input1, input2;
   int4 indices1, indices2;
   int4 comp, add;
   uint global_start, global_offset;

   add = (int4)(4, 5, 6, 7);

   /* Determine location of data in global memory */
   global_start = (get_group_id(0) + (get_group_id(0)/stage)*stage) *
                   get_local_size(0) + get_local_id(0);
   global_offset = stage * get_local_size(0);

   /* Perform swap */
   input1 = g_data[global_start];
   input2 = g_data[global_start + global_offset];
   
   indices1 = g_indices[global_start];
   indices2 = g_indices[global_start + global_offset];
   
   
   comp = (input1 < input2 ^ dir) * 4 + add;
   g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
   g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
   
   g_indices[global_start] = shuffle2(indices1, indices2, as_uint4(comp));
   g_indices[global_start + global_offset] = shuffle2(indices2, indices1, as_uint4(comp));
}

/* Perform final step of the bitonic merge */
__kernel void bsort_merge_last(__global double4 *g_data, __local double4 *l_data, int dir, __global int4* g_indices, __local int4* l_indices) {

   uint id, global_start, stride;
   double4 input1, input2, temp;
   int4 indices1, indices2, temp_indices;
   int4 comp;

   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(4, 5, 6, 7);

   /* Determine location of data in global memory */
   id = get_local_id(0);
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   /* Perform initial swap */
   input1 = g_data[global_start];
   input2 = g_data[global_start + get_local_size(0)];
   
   indices1 = g_indices[global_start];
   indices2 = g_indices[global_start + get_local_size(0)];
   
   comp = (input1 < input2 ^ dir) * 4 + add3;
   l_data[id] = shuffle2(input1, input2, as_uint4(comp));
   l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));
   
   l_indices[id] = shuffle2(indices1, indices2, as_uint4(comp));
   l_indices[id + get_local_size(0)] = shuffle2(indices2, indices1, as_uint4(comp));
   

   /* Perform bitonic merge */
   for(stride = get_local_size(0)/2; stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data[id], l_data[id + stride], dir, l_indices[id], l_indices[id+stride])
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   input1 = l_data[id]; 
   input2 = l_data[id+1];
   
   indices1 = l_indices[id];
   indices2 = l_indices[id+1];
   
   temp = input1;
   temp_indices = indices1;
   
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));
   
   indices1 = shuffle2( indices1, indices2, as_uint4(comp));
   indices2 = shuffle2( indices2, temp_indices, as_uint4(comp));
   
   VECTOR_SORT(input1, dir, indices1);
   VECTOR_SORT(input2, dir, indices2);

   /* Store the result to global memory */
   g_data[global_start + get_local_id(0)] = input1;
   g_data[global_start + get_local_id(0) + 1] = input2;
   
   g_indices[global_start + get_local_id(0)] = indices1;
   g_indices[global_start + get_local_id(0) + 1] = indices2;
   
}
