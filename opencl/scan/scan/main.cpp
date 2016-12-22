#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>

using Array = std::vector<float>;

auto constexpr SCAN_SOURCE_FILE = "scan.cl";
auto constexpr LOCAL_SCAN_KERNEL_NAME = "local_scan";
auto constexpr ADD_BOUNDS_KERNEL_NAME = "add_lefter_bounds";
size_t constexpr WORK_GROUP_SIZE = 256;

void readArray(std::istream & stream, Array & array, size_t size)
{
  for (size_t i = 0; i < size; ++i)
    stream >> array[i];
}

void writeArray(std::ostream & stream, Array const & array, size_t size)
{
  stream << std::fixed << std::setprecision(3);
  for (size_t i = 0; i < size; ++i)
    stream << array[i] << " ";
  stream << std::endl;
}

void runDeviceScan(Array const & input, Array & output)
{
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  cl::Program program;

  try {
    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    cl::Context context(devices);
    cl::CommandQueue cmdQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    std::ifstream sourceFile(SCAN_SOURCE_FILE);
    std::string sourceStr(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    cl::Program::Sources source(1, { sourceStr.c_str(), sourceStr.length() + 1 });
    program = cl::Program(context, source);
    program.build(devices);

    size_t boundValuesCount = input.size() / WORK_GROUP_SIZE;
    cl::Buffer devInput(context, CL_MEM_READ_ONLY, sizeof(float) * input.size());
    cl::Buffer devBoundValues(context, CL_MEM_READ_WRITE, sizeof(float) * boundValuesCount);
    cl::Buffer devOutput(context, CL_MEM_READ_WRITE, sizeof(float) * output.size());

    cmdQueue.enqueueWriteBuffer(devInput, CL_TRUE, 0, sizeof(float) * input.size(), &input[0]);

    cl::Kernel localScanKernel(program, LOCAL_SCAN_KERNEL_NAME);
    cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::LocalSpaceArg, cl::LocalSpaceArg> localScan(localScanKernel);
    cl::EnqueueArgs scanArgs(cmdQueue, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(WORK_GROUP_SIZE));
    size_t localSize = std::min(input.size(), WORK_GROUP_SIZE);
    localScan(scanArgs, devInput, devOutput, devBoundValues, cl::Local(sizeof(float) * localSize), cl::Local(sizeof(float) * localSize));

    cl::Kernel addBoundsKernel(program, ADD_BOUNDS_KERNEL_NAME);
    cl::make_kernel<cl::Buffer &, cl::Buffer &> addBounds(addBoundsKernel);
    cl::EnqueueArgs addBoundsArgs(cmdQueue, cl::NullRange, cl::NDRange(output.size()), cl::NDRange(WORK_GROUP_SIZE));
    addBounds(addBoundsArgs, devOutput, devBoundValues);

    cmdQueue.enqueueReadBuffer(devOutput, CL_TRUE, 0, sizeof(float) * output.size(), &output[0]);
  } catch (cl::Error const & e) {
    if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
      size_t log_size;
      clGetProgramBuildInfo(program(), devices[0](), CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program(), devices[0](), CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
      printf("%s\n", &log[0]);
    } else {
      std::cerr << "FAILED: " << e.what() << " : " << e.err() << std::endl;
    }
  }
}

size_t roundUp(size_t n) {
  return n % WORK_GROUP_SIZE
          ? n + (WORK_GROUP_SIZE - (n % WORK_GROUP_SIZE))
          : n;
}

int main()
{
  size_t n;
  std::fstream in("input.txt");
  in >> n;
  size_t size = roundUp(n);

  Array input(size, 0);
  readArray(in, input, n);

  Array output(size, 1);
  runDeviceScan(input, output);

  std::ofstream out("output.txt");
  writeArray(out, output, n);

  return 0;
}
