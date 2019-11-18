all: ibex_fileout_adarsh

ibex_fileout_adarsh: ibex_fileout_adarsh.cpp
	clang++ `pkg-config ibex --libs --cflags` -lVibes $< -o $@

ibex_stack_intersection: ibex_stack_intersection.cpp
	clang++ `pkg-config ibex --libs --cflags` $< -o $@ -std=c++11
