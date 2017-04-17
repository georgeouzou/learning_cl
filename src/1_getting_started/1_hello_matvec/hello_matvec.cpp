#include <vector>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <string>

#include <CL/cl.h>

#define PROGRAM_FILE "hello_matvec.cl"
#define KERNEL_FUNC "matvec_mult"

std::string getWorkingDirectory(const std::string& file)
{
	size_t found = file.find_last_of("/\\");
	if (found == std::string::npos) return "";
#if defined(WIN32)
	return file.substr(0, found) + "\\";
#else
	return file.substr(0, found) + "/";
#endif
}

void initData(float *mat, float *vec,  
				float *correct, size_t size)
{
	size_t i;
	for(i = 0; i < 4 * size; ++i) {
		mat[i] = i * 2.0f;
	}
	for(i = 0; i < size; ++i) {
		vec[i] = i * 3.0f;
		correct[0] += mat[i]    * vec[i];
		correct[1] += mat[i+4]  * vec[i];
		correct[2] += mat[i+8]  * vec[i];
		correct[3] += mat[i+12] * vec[i];
	}
}

cl_program createProgramFromFile(const std::string& path, cl_context ctx)
{
	cl_program pg;
	cl_int err;
	std::ifstream f(path);
	std::string code;

	if (!f.is_open()) return NULL;

	// reserve string memory
	f.seekg(0, std::ios::end);
	code.reserve(f.tellg());
	f.seekg(0, std::ios::beg);

	code.assign(
		(std::istreambuf_iterator<char>(f)),
		 std::istreambuf_iterator<char>()
	);

	const char *sources[] = { code.c_str() };
	size_t sizes[] = { code.size() };
	pg = clCreateProgramWithSource(ctx, 1, sources, sizes, &err);
	if (err != CL_SUCCESS) { return NULL; }

	return pg;
}

int main(int argc, char **argv)
{
	cl_int i, err;
	float mat[16], vec[4], result[4];
	float correct[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	
	if (argc < 1) { return -1; }
	std::string kernelfile = getWorkingDirectory(argv[0]) + PROGRAM_FILE;

	initData(mat, vec, correct, 4);

	cl_platform_id platform;
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err != CL_SUCCESS) {
		std::cerr << "cannot find any OpenCL platform" << std::endl;
	}

	cl_device_id device;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err != CL_SUCCESS) {
		std::cerr << "cannot find any OpenCL supporting GPU device" << std::endl;
	}

	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		std::cerr << "cannot create context" << std::endl;
	}

	cl_program program = createProgramFromFile(kernelfile, context);
	if (!program) {
		std::cerr << "cannot create program from source file" << std::endl;
	}
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	cl_kernel kernel = clCreateKernel(program, KERNEL_FUNC, &err);
	if (err != CL_SUCCESS) {
		std::cerr << "cannot create kernel" << std::endl;
	}
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

	if (err != CL_SUCCESS) {
		std::cerr << "cannot create command queue" << std::endl;
	}

	// create the buffers for the data 
	cl_mem mat_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(float)*16, mat, &err);
	cl_mem vec_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float)*4, vec, &err);
	cl_mem res_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, &err);
	if (err != CL_SUCCESS || !mat_buf || !vec_buf || !res_buf) {
		std::cerr << "cannot create memory buffers" << std::endl;
	} 

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buf);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_buf);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buf);

	const size_t workUnits = 4;
	clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
		&workUnits, NULL, 0, NULL, NULL);

	clEnqueueReadBuffer(queue, res_buf, CL_TRUE, 0,
		sizeof(float)*4, result, 0, NULL, NULL);

	err = 0;
	for (i = 0; i < 4; ++i) { 
		if (result[i] != correct[i]) { err = 1; }
		printf("result[%d] = %f, correct[%d] = %f\n", i, result[i], i, correct[i]);
	}
	printf("Matrix-vector multiplication %ssuccessful.\n", err ? "un" : "");
	
	clReleaseMemObject(mat_buf);
	clReleaseMemObject(vec_buf);
	clReleaseMemObject(res_buf);
	clReleaseCommandQueue(queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseContext(context);		

	return 0;
}

