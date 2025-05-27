# Find Catch2 (CMake >= 3.11 recommended)
find_package(Catch2 2.13.8 REQUIRED)
include(CTest)
include(Catch)

# Explicitly set Catch2 library path
set(CATCH2_LIBRARY_PATH "/usr/local/lib/libCatch2Main.a")
if(NOT EXISTS ${CATCH2_LIBRARY_PATH})
    message(FATAL_ERROR "Catch2 library not found at ${CATCH2_LIBRARY_PATH}")
endif()

add_executable(tester
        catalog/TestCatalog.cpp
        catalog/TestTypes.cpp
        catalog/TestMetaData.cpp
        catalog/TestCatalogEntries.cpp
        runtime/TestUTF8.cpp
        storage/TestStorage.cpp
        utility/TestSerialization.cpp
)
target_link_libraries(tester PRIVATE ${CATCH2_LIBRARY_PATH} ${LIBS} utility catalog MLIRSQLFrontend runtime runner)
set_target_properties(tester PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

enable_testing()
add_test(NAME UnitTests COMMAND tester)

catch_discover_tests(tester)
