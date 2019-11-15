all: ibex_fileout_adarsh

ibex_fileout_adarsh: ibex_fileout_adarsh.cpp
	clang++ `pkg-config ibex --libs --cflags` -lVibes $< -o $@
