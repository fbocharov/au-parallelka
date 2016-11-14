#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

using Matrix = std::vector<float>;

auto constexpr CONVOLUTION_SOURCE_FILE = "convolve.cl";
auto constexpr CONVOLUTION_KERNEL_NAME = "convolve";

void readMatrix(std::istream & stream, Matrix & matrix, size_t size) 
{
    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j)
            stream >> matrix[i * size + j];
}

void writeMatrix(std::ostream & stream, Matrix const & matrix, size_t size) 
{
    stream << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j)
            stream << matrix[i * size + j] << " ";
        stream << std::endl;
    }
}

void runDeviceConvolution(Matrix const & a, size_t aSize, Matrix const & b, size_t bSize, Matrix & result) 
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Program program;

    try {
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        
        cl::Context context(devices);
        cl::CommandQueue cmdQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
        
        std::ifstream sourceFile(CONVOLUTION_SOURCE_FILE);
        std::string sourceStr(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        
        cl::Program::Sources source(1, { sourceStr.c_str(), sourceStr.length() + 1 });
        program = cl::Program(context, source);
        program.build(devices);

        cl::Buffer devInput(context, CL_MEM_READ_ONLY, sizeof(float) * aSize * aSize);
        cl::Buffer devMask(context, CL_MEM_READ_ONLY, sizeof(float) * bSize * bSize);
        cl::Buffer devOutput(context, CL_MEM_WRITE_ONLY, sizeof(float) * aSize * aSize);

        cmdQueue.enqueueWriteBuffer(devInput, CL_TRUE, 0, sizeof(float) * aSize * aSize, &a[0]);
        cmdQueue.enqueueWriteBuffer(devMask, CL_TRUE, 0, sizeof(float) * bSize * bSize, &b[0]);

        cl::Kernel kernel(program, CONVOLUTION_KERNEL_NAME);
        cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::Buffer &, int, int> convolve(kernel);
        size_t size = aSize % 16 == 0 ? aSize : aSize + (16 - aSize % 16);
        cl::EnqueueArgs args(cmdQueue, cl::NullRange, cl::NDRange(size, size), cl::NDRange(16, 16));

        convolve(args, devInput, devMask, devOutput, bSize, aSize);

        cmdQueue.enqueueReadBuffer(devOutput, CL_TRUE, 0, sizeof(float) * aSize * aSize, &result[0]);
    } catch (cl::Error const & e) {
        std::cerr << "FAILED: " << e.what() << " : " << e.err() << std::endl;
    }
}

int main()
{
    size_t n;
    size_t m;
    std::fstream input("input.txt");
    input >> n >> m;

    Matrix A(n * n, 1);
    readMatrix(input, A, n);

    Matrix B(m * m, 1.);
    readMatrix(input, B, m);

    Matrix result(n * n, 1);
    runDeviceConvolution(A, n, B, m, result);

    std::ofstream out("output.txt");
    writeMatrix(out, result, n);

    return 0;
}
