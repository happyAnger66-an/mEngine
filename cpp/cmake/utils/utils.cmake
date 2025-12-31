function(set_architecture_definitions target)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_definitions(${target} PRIVATE
            X86_64
            ARCH_64BIT=1
            TARGET_ARCH="x64"
        )
    endif()
endfunction()