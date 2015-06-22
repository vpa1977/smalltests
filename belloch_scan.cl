#define WARP_SHIFT 4
#define GRP_SHIFT 8
#define BANK_OFFSET(n)     ((n) >> WARP_SHIFT + (n) >> GRP_SHIFT)
// https://software.intel.com/en-us/articles/using-opencl-20-work-group-functions
// num_groups == num_bins
__kernel void belloch_hist(__global double* selection, __global uint* gHistArray,
                              __global uint* gPrefixsumArray,
                              uint bin_size, 
                              uint digit_pos,
                              ,__local uint* temp
                            )
{
    int lid = get_local_id(0);
    uint binId = get_group_id(0);
    int n = get_local_size(0) * 2;

    uint group_offset = binId * bin_size;
    uint maxval = 0;
    do
    {
    	// 
        // calculate array indices and offsets to avoid SLM bank conflicts
        int ai = lid;
        int bi = lid + (n>>1);  
        int bankOffsetA = BANK_OFFSET(ai);
        int bankOffsetB = BANK_OFFSET(bi);

        // load input into local memory
        temp[ai + bankOffsetA] = gHistArray[group_offset + ai];
        temp[bi + bankOffsetB] = gHistArray[group_offset + bi];

        // parallel prefix sum up sweep phase
        int offset = 1;
        for (int d = n>>1; d > 0; d >>= 1)
        {   
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < d)
            {  
                int ai = offset * (2*lid + 1)-1;
                int bi = offset * (2*lid + 2)-1;
                ai += BANK_OFFSET(ai);
                bi += BANK_OFFSET(bi);
                temp[bi] += temp[ai];  
            }  
            offset <<= 1; 
        }

        // clear the last element
        if (lid == 0)
        {
            temp[n - 1 + BANK_OFFSET(n - 1)] = 0;
        }

        // down sweep phase
        for (int d = 1; d < n; d <<= 1)
        {  
            offset >>= 1;  
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lid < d)
            {
                int ai = offset * (2*lid + 1)-1;
                int bi = offset * (2*lid + 2)-1;
                ai += BANK_OFFSET(ai);
                bi += BANK_OFFSET(bi);
         
                uint t = temp[ai];  
                temp[ai] = temp[bi];  
                temp[bi] += t;   
            }
        }  
        barrier(CLK_LOCAL_MEM_FENCE);

        //output scan result to global memory
        gPrefixsumArray[group_offset + ai] = temp[ai + bankOffsetA] + maxval;
        gPrefixsumArray[group_offset + bi] = temp[bi + bankOffsetB] + maxval;

        //update cumulative prefix sum and shift offset for next iteration
        maxval += temp[n - 1 + BANK_OFFSET(n - 1)] + gHistArray[group_offset + n - 1];
        group_offset += n;
    }
    while(group_offset < (binId+1) * bin_size);
}