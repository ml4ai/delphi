SUFFIX:=$(shell python3-config --extension-suffix)
CFLAGS:=$(shell pkg-config libgvc --cflags)
CFLAGS:=$(shell pkg-config sqlite3 --cflags)
LDFLAGS+=$(shell pkg-config libgvc --libs)
LDFLAGS+=$(shell pkg-config sqlite3 --libs)

all: AnalysisGraph$(SUFFIX)

AnalysisGraph$(SUFFIX): wrapper.cpp AnalysisGraph.cpp Analysisgraph.hpp
	c++ -Wall -shared -std=c++17 \
	  $(LDFLAGS) $(CFLAGS)\
	  -undefined dynamic_lookup `python3 -m pybind11 --includes`\
	  $< -o $@
clean:
	rm AnalysisGraph$(SUFFIX)
