cmake_minimum_required(VERSION 3.7)
project(sds_core LANGUAGES CXX CUDA)
#cmake_policy(SET CMP0057 NEW)
enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 20)


find_package(Eigen3)
#set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G") 
#set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)

#add_executable(vortex src/vortex_perch_mppi.cu src/vortex_model.cu src/flat_plate_glider.cu src/mppi.cu)
#target_include_directories(vortex PRIVATE include)
#target_include_directories(vortex SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
#target_link_libraries(vortex cublas)
#target_compile_options(vortex PRIVATE CUDA_FLAGS "-diag-suppress 20012" )

#add_executable(rollout src/rollout_generator.cu src/vortex_model.cu src/flat_plate_glider.cu)
#target_include_directories(rollout PRIVATE include)
#target_include_directories(rollout SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
#target_link_libraries(rollout cublas)

#add_executable(main src/main.cu src/vortex_model.cu src/flat_plate_glider.cu src/mppi.cu)
#target_include_directories(main PRIVATE include)
#target_include_directories(main SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR} ${SNOPT_INCLUDE_DIR})
#target_link_libraries(main cublas ${SNOPT_LIBRARIES})

#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#add_library(glider_dynamics STATIC src/flat_plate_glider.cu)
#target_include_directories(glider_dynamics PRIVATE include)
#set_target_properties(glider_dynamics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(glider_dynamics PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

#add_executable(tvlqr src/tvlqr.cu src/vortex_model.cu src/flat_plate_glider.cu src/mppi.cu)
#enable_language(CUDA)
#target_include_directories(tvlqr PRIVATE include)
#target_include_directories(tvlqr SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
#target_link_libraries(tvlqr cublas)

add_library(sds_core STATIC src/sds_core.cpp)
#add_library(sds_core INTERFACE)
target_include_directories(sds_core PUBLIC include ${EIGEN3_INCLUDE_DIR})
#target_link_libraries(sds_core INTERFACE cublas)

set_target_properties(sds_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)


add_executable(dubin_cem example/dubin_cem.cu)
#target_include_directories(dubin_cem PRIVATE include)
#target_include_directories(dubin_cem SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(dubin_cem sds_core)

target_compile_options(dubin_cem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)


#set(PYBIND11_FINDPYTHON ON)
#find_package(pybind11 CONFIG REQUIRED)
#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#pybind11_add_module(vortex_pybinds src/pybinds.cu src/vortex_model.cu src/flat_plate_glider.cu src/tvlqr.cu src/mppi.cu)
#target_link_libraries(vortex_pybinds PRIVATE cublas)
#target_include_directories(vortex_pybinds SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
#target_include_directories(vortex_pybinds PRIVATE include)
#set(Stubgen_Executable "/usr/bin/stubgen")
#add_custom_command(TARGET vortex_pybinds POST_BUILD
#    COMMAND ${Stubgen_Executable} -m vortex_pybinds -o .
#    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#    COMMENT "Use stubgen to create .pyi for statement completion")
#
#set(PYI_FILE "${CMAKE_CURRENT_BINARY_DIR}/vortex_pybinds.pyi")


#install(TARGETS vortex_pybinds 
#        LIBRARY DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)
#execute_process(
#  COMMAND ${Python3_EXECUTABLE} -c
#  "import site; print(site.getsitepackages()[0])"
#  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
#  OUTPUT_STRIP_TRAILING_WHITESPACE
#)
#install(TARGETS vortex_pybinds
#        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})

#if(EXISTS "${PYI_FILE}")
#    #install(FILES "${PYI_FILE}"
#    #    DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
#    #)
#    #install(FILES "${PYI_FILE}"
#    #    DESTINATION ${PYTHON_SITE_PACKAGES}
#    #)
#else()
#    message(WARNING "PYI file ${PYI_FILE} not found, skipping installation")
#endif()
#