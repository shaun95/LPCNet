cmake_minimum_required (VERSION 3.0)

PROJECT(LPCNet)

#SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -W -Wextra -Wno-unused-function -O3 -g")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wextra -Wno-unused-function -O3 -g")

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# ARM specific flags
FIND_PACKAGE(ARM)
IF (ASIMD_FOUND)
  MESSAGE(STATUS "asimd/Neon found with compiler flag : -D__ARM_NEON__")
  SET(CMAKE_C_FLAGS "-D__ARM_NEON__ ${CMAKE_C_FLAGS}")
ELSEIF (NEON_FOUND)
  MESSAGE(STATUS "Neon found with compiler flag : -mfpu=neon -D__ARM_NEON__")
  SET(CMAKE_C_FLAGS "-mfpu=neon -D__ARM_NEON__ ${CMAKE_C_FLAGS}")
ENDIF (ASIMD_FOUND)
IF (CORTEXA8_FOUND)
  MESSAGE(STATUS "Cortex-A8 Found with compiler flag : -mcpu=cortex-a8")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a8 -fprefetch-loop-arrays ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA8_FOUND)
IF (CORTEXA9_FOUND)
  MESSAGE(STATUS "Cortex-A9 Found with compiler flag : -mcpu=cortex-a9")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a9 ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA9_FOUND)

FIND_PACKAGE(SSE) # checks SSE, AVX and AVX2
IF(C_SSE2_FOUND)
  MESSAGE(STATUS "SSE2 Found")
  SET(CMAKE_C_FLAGS "${C_SSE2_FLAGS} -DUSE_SSE2 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE2_FOUND)
IF(C_SSE3_FOUND)
  MESSAGE(STATUS "SSE3 Found")
  SET(CMAKE_C_FLAGS "${C_SSE3_FLAGS} -DUSE_SSE3 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE3_FOUND)
# we dont set -mavx and -mavx2 flags globally, but only for specific files
# however, we want to enable the AVX codepaths, so we still need to
# add USE_AVX and USE_AVX2 macro defines
IF(C_AVX_FOUND)
  MESSAGE(STATUS "AVX Found")
  SET(CMAKE_C_FLAGS "-mavx -D__AVX__ ${CMAKE_C_FLAGS}")
ENDIF(C_AVX_FOUND)
IF(C_AVX2_FOUND)
  MESSAGE(STATUS "AVX2 Found")
  SET(CMAKE_C_FLAGS "-mavx2 -mfma -D__AVX2__ ${CMAKE_C_FLAGS}")
ENDIF(C_AVX2_FOUND)


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/library/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/library/src)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

ADD_LIBRARY(acoustic SHARED library/src/freq.c library/src/kiss_fft.c library/src/pitch.c library/src/celt_lpc.c)
TARGET_LINK_LIBRARIES(acoustic m)

ADD_EXECUTABLE(dump_data library/src/dump_data.c)
TARGET_LINK_LIBRARIES(dump_data acoustic)


ADD_EXECUTABLE(test_vec library/src/test_vec.c)
TARGET_LINK_LIBRARIES(test_vec acoustic)

IF(WITH_MODEL_DATA)
	MESSAGE(STATUS "WITH_MODEL_DATA ON. The test lpcnet of c/cpp version will be exported!")
  ADD_EXECUTABLE(test_lpcnet library/src/test_lpcnet.c library/src/lpcnet.c library/src/nnet.c library/src/nnet_data.c)
  TARGET_LINK_LIBRARIES(test_lpcnet acoustic)
ELSE()
	MESSAGE(STATUS "WITH_MODEL_DATA OFF. The test lpcnet of c/cpp version will NOT be exported!")
ENDIF()

IF(WITH_PYTHON_PATCH)
  ADD_SUBDIRECTORY(pybind11)
  PYBIND11_ADD_MODULE(LPCNet library/src/extension.cpp)
  TARGET_LINK_LIBRARIES(LPCNet PRIVATE acoustic)
ENDIF()
