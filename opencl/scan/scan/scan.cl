#define SWAP(a,b) {__local float * tmp=a; a=b; b=tmp;}

__kernel void local_scan(__global float * input, __global float * output, __global float * bound_values, 
  __local float * a, __local float * b)
{
  uint lid = get_local_id(0);
  uint gid = get_group_id(0);
  uint block_size = get_local_size(0);

  uint real_index = lid + gid * block_size;

  a[lid] = b[lid] = input[real_index];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint s = 1; s < block_size; s *= 2) {
    if (lid > (s - 1))
      b[lid] = a[lid] + a[lid - s];
    else
      b[lid] = a[lid];

    barrier(CLK_LOCAL_MEM_FENCE);
    SWAP(a, b);
  }

  output[real_index] = a[lid];
  if (lid == block_size - 1)
    bound_values[gid] = a[lid];
}

__kernel void add_lefter_bounds(__global float * input, __global float * bound_values)
{
  uint lid = get_local_id(0);
  uint gid = get_group_id(0);
  uint block_size = get_local_size(0);

  if (gid > 0)
    for (uint i = 0; i < gid; ++i)
      input[lid + gid * block_size] += bound_values[i];
}
