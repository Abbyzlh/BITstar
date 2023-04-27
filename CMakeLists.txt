cmake_minimum_required(VERSION 3.22)
project(BIT VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

#set(${CMAKE_BINARY_DIR} "build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)




add_subdirectory(${PROJECT_SOURCE_DIR}/src)
target_include_directories(${PROJECT_NAME} PUBLIC "include")

target_link_libraries(${PROJECT_NAME} PUBLIC BatchInformedTree)
target_link_libraries(${PROJECT_NAME} PUBLIC WorkingSpace)

