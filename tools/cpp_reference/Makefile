# Build the golden-data harness against the (read-only) cco sources.
# Flags and libraries mirror /home/sd2402/ccoptimization/cco/CCOptimization/makefile
# (DELLA branch); main.cpp is excluded, nothing is written inside CCO_DIR.
CCO_DIR ?= /home/sd2402/ccoptimization/cco
CXX     ?= g++
CXXFLAGS = -Wall -std=c++11 -O3 -fopenmp -DDELLA -I$(CCO_DIR)
LIBS     = -lgsl -lgslcblas -lnlopt -lm
BUILD    = build
CCO_SRC  = GeometryVector RandomGenerator etc PeriodicCellList StructureFactor \
           PairCorrelation Potential StructureOptimization \
           StructureOptimizationAdvanced MD_System Trajectories
OBJS     = $(addprefix $(BUILD)/,$(addsuffix .o,$(CCO_SRC))) $(BUILD)/dump_reference.o
TARGET   = $(BUILD)/dump_reference

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(OBJS) -fopenmp $(LIBS)

$(BUILD)/dump_reference.o: dump_reference.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD)/%.o: $(CCO_DIR)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

flags:
	@echo "$(CXX) $(CXXFLAGS) $(LIBS)"

clean:
	rm -rf $(BUILD)

.PHONY: all clean flags
