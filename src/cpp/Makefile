bgl_example: bgl_example.cpp
	c++ -std=c++17 $< -o $@ -I/opt/local/include

pymodule: bgl_example.cpp
	c++ -O3 -Wall -shared -std=c++17 -I/opt/local/include \
	  -undefined dynamic_lookup `python3 -m pybind11 --includes`\
	  bgl_example.cpp -o example`python3-config --extension-suffix`

