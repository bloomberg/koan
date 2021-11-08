CXX = g++

BUILD_PATH = ./build
EIGEN_INCLUDE = ./eigen/

INCLUDES = -I$(EIGEN_INCLUDE) -I./

CXXFLAGS = -std=c++17 -Wall -Wextra -pthread

ZIPFLAGS =  -lz -DKOAN_ENABLE_ZIP

OPTFLAGS = -Ofast -march=native -mtune=native
DEBUGFLAGS = -g -O0

build_path:
	@mkdir -p $(BUILD_PATH)

% : %.cpp build_path
	$(CXX) $< $(CXXFLAGS) ${ZIPFLAGS} $(OPTFLAGS) $(INCLUDES) -o $(BUILD_PATH)/$@

debug : koan.cpp build_path
	$(CXX) $< $(CXXFLAGS) ${ZIPFLAGS} $(DEBUGFLAGS) $(INCLUDES) -o $(BUILD_PATH)/koan

test_utils : tests/test_utils.cpp build_path
	$(CXX) $< $(CXXFLAGS) ${ZIPFLAGS} $(DEBUGFLAGS) $(INCLUDES) -I./extern/ -o $(BUILD_PATH)/test_utils

test_gradcheck : tests/test_gradcheck.cpp build_path
	$(CXX) $< $(CXXFLAGS) ${ZIPFLAGS} $(DEBUGFLAGS) $(INCLUDES) -I./extern/ -o $(BUILD_PATH)/test_gradcheck

all: koan test_utils test_gradcheck

clean:
	rm -rf $(BUILD_PATH)
