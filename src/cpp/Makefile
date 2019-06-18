SUFFIX:=$(shell python3-config --extension-suffix)

all: AnalysisGraph$(SUFFIX)

AnalysisGraph$(SUFFIX): AnalysisGraph.cpp
	c++ -O3 -Wall -shared -std=c++17 -I/opt/local/include \
	  -undefined dynamic_lookup `python3 -m pybind11 --includes`\
	  $< -o $@
