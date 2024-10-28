CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -Wall $(shell python3 -m pybind11 --includes)
PY_LDFLAGS = $(shell python3-config --ldflags) -lopenblas -shared -fPIC

all: linalg

linalg: bindings.o LinearAlgebra.o
	$(CXX) $^ -o linalg_core`python3-config --extension-suffix` $(PY_LDFLAGS) $(CXXFLAGS)

bindings.o: bindings.cpp LinearAlgebra.h
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

LinearAlgebra.o: LinearAlgebra.cpp LinearAlgebra.h
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

clean:
	rm -f *.o linalg_core`python3-config --extension-suffix`