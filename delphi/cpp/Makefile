SUFFIX:=$(shell python3-config --extension-suffix)
CFLAGS:=$(shell pkg-config sqlite3 --cflags)
LDFLAGS+=$(shell pkg-config sqlite3 --libs)
extension=extension$(SUFFIX)

all: $(extension)

$(extension): wrapper.cpp AnalysisGraph.cpp AnalysisGraph.hpp
	c++ -O3 -Wall -shared -std=c++17 \
	  $(LDFLAGS) $(CFLAGS)\
	  -fPIC \
	  `python3 -m pybind11 --includes`\
	  $< -o $(extension)
clean:
	rm $(extension)
