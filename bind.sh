g++ -O3 -Wall -g -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) \
	arch_presser/ray_sum.cpp -o arch_presser/ray_sum$(python3-config --extension-suffix)
