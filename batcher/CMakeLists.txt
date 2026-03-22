cmake_minimum_required(VERSION 3.10)
project(SchedulerCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB_RECURSE HEADERS "include/*.h")

file(GLOB_RECURSE SOURCES "src/*.cpp")

source_group("Header Files" FILES ${HEADERS})
source_group("Source Files" FILES ${SOURCES})

add_executable(scheduler ${SOURCES} ${HEADERS})

target_include_directories(scheduler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)