
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics  : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define VALUE_TYPE double
#define COMPAT_VALUE_TYPE atomic_double

void my_atomic_add(__global double * loc, const double f) 
{   
	VALUE_TYPE old = *loc;   
	VALUE_TYPE sum = old + f;   
	volatile bool test =true;
	while ((test = atomic_compare_exchange_weak( (COMPAT_VALUE_TYPE*)loc, &old,sum)) ==false)  
		sum = old + f;
} 


__kernel void test_atomic_lock(__global double* output )
{
	int idx = get_global_id(0);
	
	my_atomic_add(output, 1);
	
}
