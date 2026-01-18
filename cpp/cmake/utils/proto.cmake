# This file was designed for new directory structure exemplified here:
# https://git.xiaojukeji.com/av/experimental_layout to add path to proto files
# and generate proto libraries.
#
#find_package(OpenSSL REQUIRED)
find_package(Protobuf REQUIRED)
find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

include_guard(GLOBAL)

function(add_protobuf_target TARGET_NAME)
    set(options GENERATE_PYTHON)
    set(oneValueArgs OUTPUT_DIR NAMESPACE PYTHON_OUTPUT_DIR)
    set(multiValueArgs PROTO_FILES IMPORT_DIRS)
    
    cmake_parse_arguments(
        ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
    )
    
    if(NOT ARG_PROTO_FILES)
        message(FATAL_ERROR "No PROTO_FILES specified for target ${TARGET_NAME} ")
    endif()
   
    if(NOT ARG_OUTPUT_DIR)
        set(ARG_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
    endif()
        
    if(NOT EXISTS "${ARG_OUTPUT_DIR}")
        file(MAKE_DIRECTORY "${ARG_OUTPUT_DIR}")
    endif()
    
    # 设置 Python 输出目录
    if(ARG_GENERATE_PYTHON)
        if(NOT ARG_PYTHON_OUTPUT_DIR)
            set(ARG_PYTHON_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/python)
        endif()
        
        if(NOT EXISTS "${ARG_PYTHON_OUTPUT_DIR}")
            file(MAKE_DIRECTORY "${ARG_PYTHON_OUTPUT_DIR}")
        endif()
        
        # 创建 __init__.py 文件以使其成为 Python 包
        set(PYTHON_INIT_FILE "${ARG_PYTHON_OUTPUT_DIR}/__init__.py")
        if(NOT EXISTS "${PYTHON_INIT_FILE}")
            file(WRITE "${PYTHON_INIT_FILE}" "# Generated Protobuf Python package\n")
        endif()
    endif()
    
    # 创建库目标
    add_library(${TARGET_NAME} SHARED)
    
    # 为每个 .proto 文件生成代码
    set(GENERATED_SOURCES)
    set(GENERATED_HEADERS)
    set(GENERATED_PYTHON_FILES)
    
    foreach(PROTO_FILE ${ARG_PROTO_FILES})
        get_filename_component(PROTO_ABS ${PROTO_FILE} ABSOLUTE)
        get_filename_component(PROTO_NAME_WE ${PROTO_FILE} NAME_WE)
        
        # 设置生成文件路径
        set(GENERATED_SRC "${ARG_OUTPUT_DIR}/${PROTO_NAME_WE}.pb.cc")
        set(GENERATED_HDR "${ARG_OUTPUT_DIR}/${PROTO_NAME_WE}.pb.h")
        
        # 设置 Python 生成文件路径
        if(ARG_GENERATE_PYTHON)
            set(GENERATED_PY "${ARG_PYTHON_OUTPUT_DIR}/${PROTO_NAME_WE}_pb2.py")
            list(APPEND GENERATED_PYTHON_FILES ${GENERATED_PY})
        endif()
        
        # 构建 protoc 命令参数
        set(PROTOC_ARGS
            --cpp_out=${ARG_OUTPUT_DIR}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        )
        
        # 添加 Python 输出参数
        if(ARG_GENERATE_PYTHON)
            list(APPEND PROTOC_ARGS --python_out=${ARG_PYTHON_OUTPUT_DIR})
        endif()
        
        # 添加导入目录
        foreach(IMPORT_DIR ${ARG_IMPORT_DIRS})
            list(APPEND PROTOC_ARGS --proto_path=${IMPORT_DIR})
        endforeach()
        
        # 添加 proto 文件
        list(APPEND PROTOC_ARGS ${PROTO_ABS})
        
        # 创建生成命令的输出列表
        set(OUTPUT_FILES ${GENERATED_SRC} ${GENERATED_HDR})
        if(ARG_GENERATE_PYTHON)
            list(APPEND OUTPUT_FILES ${GENERATED_PY})
        endif()
        
        # 设置注释消息
        set(COMMENT_MSG "Generating ${PROTO_NAME_WE}.pb.{cc,h}")
        if(ARG_GENERATE_PYTHON)
            set(COMMENT_MSG "Generating ${PROTO_NAME_WE} protobuf files (C++ and Python)")
        endif()
        
        # 创建生成命令
        add_custom_command(
            OUTPUT ${OUTPUT_FILES}
            COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            ARGS ${PROTOC_ARGS}
            DEPENDS ${PROTO_ABS}
            COMMENT "${COMMENT_MSG}"
            VERBATIM
        )
        
        list(APPEND GENERATED_SOURCES ${GENERATED_SRC})
        list(APPEND GENERATED_HEADERS ${GENERATED_HDR})
    endforeach()
    
    # 添加源文件到目标
    target_sources(${TARGET_NAME} PRIVATE ${GENERATED_SOURCES})
    
    # 设置包含目录
    target_include_directories(${TARGET_NAME}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:include>")
    
    # 链接 Protobuf 库
    target_link_libraries(${TARGET_NAME} PUBLIC ${Protobuf_LIBRARIES})
    
    # 设置命名空间
    #if(ARG_NAMESPACE)
    #    set_target_properties(${TARGET_NAME} PROPERTIES
    #        EXPORT_NAME ${ARG_NAMESPACE}::${TARGET_NAME}
    #    )
    #endif()

    # 将生成的文件保存到父作用域
    set(${TARGET_NAME}_GENERATED_SOURCES ${GENERATED_SOURCES} PARENT_SCOPE)
    set(${TARGET_NAME}_GENERATED_HEADERS ${GENERATED_HEADERS} PARENT_SCOPE)
    
    # 如果有生成 Python 文件，也导出到父作用域
    if(ARG_GENERATE_PYTHON)
        set(${TARGET_NAME}_GENERATED_PYTHON_FILES ${GENERATED_PYTHON_FILES} PARENT_SCOPE)
        set(${TARGET_NAME}_PYTHON_OUTPUT_DIR ${ARG_PYTHON_OUTPUT_DIR} PARENT_SCOPE)
    endif()

endfunction()