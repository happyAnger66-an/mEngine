cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

project(mEngine_protos LANGUAGES CXX)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/utils/proto.cmake)

add_protobuf_target(mEngine_protos
  GENERATE_PYTHON ON
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}
  PROTO_FILES
trt_io.proto
trt_engine.proto
cuda_buffer_config.proto
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include
)