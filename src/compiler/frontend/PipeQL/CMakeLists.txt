add_library(MLIRPipeQLFrontend
    Parser.cpp
    MLIRPipeQLVisitor.cpp
)

target_include_directories(MLIRPipeQLFrontend
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/vendored/PipeQL/src/include
    ${ANTLR4_INCLUDE_DIRS}
)

add_dependencies(MLIRPipeQLFrontend 
    ec-rt-defs 
    antlr4_static
    pipeql
)

target_link_libraries(MLIRPipeQLFrontend 
    PUBLIC 
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs}
    runtime_funcs_ptr
    utility
    catalog
    antlr4_static
    pipeql
) 