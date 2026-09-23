cmake_minimum_required(VERSION 3.20)
project(PbrIblHomework LANGUAGES CXX)

add_executable(PbrIblHomework WIN32
    src/main.cpp src/GBuffer.cpp src/GBuffer.h src/SpatialIndex.cpp src/SpatialIndex.h)
target_compile_features(PbrIblHomework PRIVATE cxx_std_17)
target_compile_definitions(PbrIblHomework PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(PbrIblHomework PRIVATE d3d12 dxgi d3dcompiler windowscodecs)

add_custom_command(TARGET PbrIblHomework POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:PbrIblHomework>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/shaders/Textured.hlsl $<TARGET_FILE_DIR:PbrIblHomework>/Textured.hlsl)
