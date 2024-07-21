output:
	g++ -std=c++17 -o output -I/home/eml/tensorflow neural_network.cpp tools.cpp libtensorflow-lite.a -pthread -lrt -ldl -lstdc++fs
all: output

clean:
	rm output