cmake_minimum_required(VERSION 3.20)
project(calibrate)
set(CMAKE_CXX_STANDARD 17)
add_executable(calibrate src/main.cpp src/calibrate.cpp)
target_include_directories(calibrate PRIVATE inc/)

find_package(OpenCV REQUIRED)
target_link_libraries(calibrate ${OpenCV_LIBS})

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(calibrate ${BLAS_LIBS})
target_link_libraries(calibrate ${LAPACK_LIBS})

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(calibrate ${ARMADILLO_LIBRARIES})
