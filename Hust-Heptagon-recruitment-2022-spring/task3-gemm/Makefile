gemm: main.cpp
	$(CXX) main.cpp -o gemm -O0 -g -Wall -Werror -std=c++11 -lm -lpthread

.PHONY: clean
clean:
	- rm -f gemm

.PHONY: run
run: 
	./gemm

.PHONY: rmdata
rmdata:
	- rm -rf data*	 
