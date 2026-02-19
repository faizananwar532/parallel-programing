<pre><div class="text_to_html">cmake_minimum_required(VERSION 3.28)
project(helloworld C)

set(CMAKE_C_STANDARD 11)

add_compile_options(-O2 -Wall -Wextra -fopenmp)

find_package(OpenMP REQUIRED)

add_executable(hello_world main.c)

target_link_libraries(hello_world PRIVATE OpenMP::OpenMP_C)</div></pre>

