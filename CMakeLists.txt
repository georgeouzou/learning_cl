cmake_minimum_required(VERSION 2.8)
project (LearnOpenCL)

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

list(APPEND CMAKE_CXX_FLAGS "-std=c++11")
	
# link with opencl library
set(OpenCL_LIBRARY OpenCL)
set(OpenCL_LIBRARY_PATH /opt/amdgpu-pro/lib/x86_64-linux-gnu/)
link_directories(${OpenCL_LIBRARY_PATH})

set(CHAPTERS
	1_getting_started
)

set(1_getting_started
	1_hello_matvec
)

# create every project file
foreach(CHAPTER ${CHAPTERS})
	string(REPLACE "_" ";" CHAPTER_TOKENS ${CHAPTER})
	list(GET CHAPTER_TOKENS 0 CHAPTER_NUMBER)
	foreach(DEMO ${${CHAPTER}})
		# create executable
		file(GLOB SOURCE
			"src/${CHAPTER}/${DEMO}/*.h"
			"src/${CHAPTER}/${DEMO}/*.cpp"
		)
		set(NAME "${CHAPTER_NUMBER}_${DEMO}")
		add_executable(${NAME} ${SOURCE})
		target_link_libraries(${NAME} ${OpenCL_LIBRARY})
		set_target_properties(${NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin/${CHAPTER}")
		# copy kernel files to output
		file(GLOB KERNELS
			"src/${CHAPTER}/${DEMO}/*.cl"
		)
		foreach(KERNEL ${KERNELS})
			file(COPY ${KERNEL} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/${CHAPTER})
		endforeach(KERNEL)
	endforeach(DEMO)
endforeach(CHAPTER)

