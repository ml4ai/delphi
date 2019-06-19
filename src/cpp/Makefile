SUFFIX:=$(shell python3-config --extension-suffix)
CFLAGS:=$(shell pkg-config libgvc --cflags)
LDFLAGS+=$(shell pkg-config libgvc --libs)

all: AnalysisGraph$(SUFFIX)

AnalysisGraph$(SUFFIX): AnalysisGraph.cpp
	c++ -O3 -Wall -shared -std=c++17 \
	  $(LDFLAGS) $(CFLAGS)\
	  -undefined dynamic_lookup `python3 -m pybind11 --includes`\
	  $< -o $@
clean:
	rm AnalysisGraph$(SUFFIX)
