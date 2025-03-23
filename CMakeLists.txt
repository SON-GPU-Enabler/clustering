cmake_minimum_required(VERSION 3.5)

find_package(CUDA REQUIRED 7.0)
find_package(OpenMP)

if(CMAKE_VERSION VERSION_LESS "3.10")
  find_package(Matlab COMPONENTS MAIN_PROGRAM)
  matlab_get_version_from_matlab_run(${Matlab_MAIN_PROGRAM} matlab_version)
  if(${matlab_version})
    if(matlab_version VERSION_LESS "8.2")
      message("-- Could NOT find Matlab: Found unsuitable version \"${matlab_version}\", but required is at least \"8.2\" (${Matlab_MAIN_PROGRAM})")
      set(MATLAB_FOUND FALSE)
    else()
      message("-- Found Matlab: Found suitable version \"${matlab_version}\", minimum required is \"8.2\"")
    endif()
  else()
    message("-- Found Matlab: Found presumably suitable version, minimum required is \"8.2\"")
  endif()
else()
  find_package(Matlab 8.2)
endif()

# setting the compiler flags for use of OpenMP
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DTHRUST_HOST_SYSTEM=THRUST_HOST_BACKEND_OMP)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_35;-std=c++11;-O3;-Xcompiler;-fPIC)

set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "-lm -lc")


if(MATLAB_FOUND)
  
  cuda_include_directories(${Matlab_INCLUDE_DIRS} /opt/B12/B12c/include)
  
  cuda_add_library(clusteringCoefficients_mex ${CMAKE_SOURCE_DIR}/clusteringCoefficients_mex.cu SHARED)
  set_target_properties(clusteringCoefficients_mex PROPERTIES PREFIX "" SUFFIX ".${Matlab_MEX_EXTENSION}")
  target_link_libraries(clusteringCoefficients_mex ${Matlab_LIBRARIES})
  
  install(TARGETS clusteringCoefficients_mex 
          RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}
          LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}
          ARCHIVE DESTINATION ${CMAKE_SOURCE_DIR})
  
endif(MATLAB_FOUND)