__kernel void convolve(__global float * input, __global float * mask, __global float * output, 
    int maskSize, int inputSize)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= inputSize || y >= inputSize)
        return;

    float result = 0;
    for (int i = 0; i < maskSize; ++i) {
        for (int j = 0; j < maskSize; ++j) {
            int row = x + i - maskSize / 2;
            int col = y + j - maskSize / 2;
            if ((0 <= row && row < inputSize) && (0 <= col && col < inputSize))
                result += input[row * inputSize + col] * mask[i * maskSize + j];
        }
    }

    output[x * inputSize + y] = result;
}
