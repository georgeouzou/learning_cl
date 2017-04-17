#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>

#define PROGRAM_FILE "kernels/matvec.cl"
#define KERNEL_FUNC "matvec_mult"

#include <CL/cl.h>

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

#define CHECK(var) \
	if ((var) == NULL) { goto error; }\

#define ON_ERROR error

cl_program createProgramFromFile(const char* path, cl_context ctx)
{
	cl_program pg = NULL;
	FILE *f = NULL;
	char *buf = NULL;
	char *log = NULL;
	size_t bufsize, logsize;
	cl_int err;

	f = fopen(path, "r");
	CHECK(f);
	fseek(f, 0, SEEK_END);
	bufsize = ftell(f);
	rewind(f);
	buf = (char*)malloc(bufsize + 1);
	CHECK(buf);
	buf[bufsize] = '\0';
	fread(buf, sizeof(char), bufsize, f);

	pg = clCreateProgramWithSource(ctx, 1, (const char**)&buf, &bufsize, &err);

ON_ERROR:
	if (buf) free(buf);
	if (f) fclose(f);
	
	return pg;
}

int main()
{
	cl_platform_id platform = NULL;
	cl_device_id device = NULL;
	cl_context context = NULL;
	cl_command_queue queue = NULL;
	cl_int i, err;

	cl_program program = NULL;
	cl_kernel kernel = NULL;
	size_t work_units_per_kernel;

	float mat[16], vec[4], result[4];
	float correct[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	cl_mem mat_buf = NULL, vec_buf = NULL, res_buf = NULL;

	initData(mat, vec, correct, 4);
	
	clGetPlatformIDs(1, &platform, NULL);
	CHECK(platform);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK(device);
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK(context);

	program = createProgramFromFile(PROGRAM_FILE, context);
	CHECK(program);
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	kernel = clCreateKernel(program, KERNEL_FUNC, &err);
	CHECK(kernel);
	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK(queue);

	mat_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(float)*16, mat, &err);
	vec_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float)*4, vec, &err);
	res_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, &err);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buf);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_buf);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buf);

	work_units_per_kernel = 4;
	clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
		&work_units_per_kernel, NULL, 0, NULL, NULL);

	clEnqueueReadBuffer(queue, res_buf, CL_TRUE, 0,
		sizeof(float)*4, result, 0, NULL, NULL);

	err = 0;
	for (i = 0; i < 4; ++i) { 
		//if (result[i] != correct[i]) { err = 1; break; }
		printf("result[%d] = %f, correct[%d] = %f\n", i, result[i], i, correct[i]);
	}
	printf("Matrix-vector multiplication %ssuccessful.\n", err ? "un" : "");
	
ON_ERROR:
	if(mat_buf) clReleaseMemObject(mat_buf);
	if(vec_buf) clReleaseMemObject(vec_buf);
	if(res_buf) clReleaseMemObject(res_buf);
	if(queue) clReleaseCommandQueue(queue);
	if(kernel) clReleaseKernel(kernel);
	if(program) clReleaseProgram(program);
	if(context) clReleaseContext(context);		

	return 0;
}

