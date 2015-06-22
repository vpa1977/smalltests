// buckethelp.cpp : Defines the entry point for the console application.

//

#include "stdafx.h"

#include <vector>

#include <limits>

#include <iostream>

void split(double* in, int* buckets, double pivot, int n_bucket, int step, int num_buckets, double base_value)

{

	if (*buckets == n_bucket) // split 1 bucket at a time

{

	long long cur_backet = (*in - base_value) / pivot;

*buckets = cur_backet + step * num_buckets;

}

}

void reduce_counts(int* bucket_assigned, int* reduce_result, int step, int num_buckets)

{

	int min_bucket = step * num_buckets;

	if (*bucket_assigned >= min_bucket)

{

++reduce_result[*bucket_assigned - min_bucket];

}

}

void copy_to_result(double* values, int* buckets, std::vector<double>& result, int step, int num_buckets, int max_bucket

	// rescan workgorup,

)

{

	int min_bucket = step* num_buckets;

	if (*buckets >= min_bucket && *buckets < max_bucket)

	result.push_back(*values);

}

　

int _tmain(int argc, _TCHAR* argv[])

{

std::vector<double> test;

	for (int i = 0; i < 1000; i++)

test.push_back(i);

std::vector<int> buckets(1000);

std::vector<double> result;

	int TARGET_K = 33;

　

	// algorithm starts here

	int K = TARGET_K;

std::vector<int> need_buckets;

	int step = 0;

	int split_bucket = 0;

	int bucket = 0;

	int num_buckets = 10;

	double base_value = 0;

std::vector<int> reduce_result(num_buckets);

	// find max to determine pivot value

	double pivot = 1000 / num_buckets;

	while (K > 0)

{

	for (int i = 0; i < 1000; ++i)

split(&test[i], &buckets[i], pivot, split_bucket, step, num_buckets, base_value);

	// reduce bucket counts

	for (int i = 0; i < reduce_result.size(); ++i)

reduce_result[i] = 0;

	for (int i = 0; i < 1000; ++i)

reduce_counts(&buckets[i], &reduce_result[0], step , num_buckets);

	// host code - analyse reduce result and schedule the next round

	for (int i = 0; i < reduce_result.size(); i++)

{

	if (reduce_result[i] > K)

{

split_bucket = i + step*num_buckets;

	break;

}

	else

{

K = K - reduce_result[i];

}

}

	// copy data to result - use workgroup scans for the reduce_count steps

	for (int i = 0; i < 1000; ++i)

copy_to_result(&test[i], &buckets[i], result, step, num_buckets, split_bucket);

base_value += pivot * (split_bucket - step*num_buckets);

	// update pivot

pivot = pivot / num_buckets;

	// increase step

step++;

}

std::cout << "Steps " << step << std::endl;

	for (int i = 0; i < TARGET_K; ++i)

std::cout << result[i] << " ";

std::cout << std::endl;

　

	return 0;

}
