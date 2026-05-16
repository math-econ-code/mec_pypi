# Build the RROA C library + smoke-test binary.
#
# Layout:
#   src/    -> dw.c, dw.h        (the library)
#   tests/  -> test_dw.c         (smoke test)
#   bench/  -> benchmark_forward.py (Python benchmark; nothing to build)
#
# Build artifacts:
#   libdw.{dylib,so}   in this directory (loaded by ../_bindings.py)
#   tests/test_dw      smoke-test binary
#   build/*.o          intermediate objects (gitignored)
#
# Targets:
#   make / make all  -> libdw + test binary
#   make lib         -> libdw only (what the Python wrapper needs)
#   make run         -> runs the smoke test
#   make clean       -> remove build artifacts
#
# OpenMP requires libomp on macOS (`brew install libomp`).

GUROBI_HOME ?= /Library/gurobi1002/macos_universal2
CC          ?= cc
CFLAGS      ?= -O3 -Wall -Wextra -std=c99 -fPIC

UNAME       := $(shell uname)
ifeq ($(UNAME),Darwin)
    SHLIB_EXT   := dylib
    SHLIB_FLAGS := -dynamiclib -install_name @rpath/libdw.dylib
    LIBOMP      ?= $(shell brew --prefix libomp 2>/dev/null)
    OMP_CFLAGS  := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
    OMP_LIBS    := -L$(LIBOMP)/lib -lomp
    OMP_RPATH   := -Wl,-rpath,$(LIBOMP)/lib
else
    SHLIB_EXT   := so
    SHLIB_FLAGS := -shared
    OMP_CFLAGS  := -fopenmp
    OMP_LIBS    := -fopenmp
    OMP_RPATH   :=
endif

INC     := -Isrc -I$(GUROBI_HOME)/include $(OMP_CFLAGS)
LIBS    := -L$(GUROBI_HOME)/lib -lgurobi100 $(OMP_LIBS) -lm
LDFLAGS := -Wl,-rpath,$(GUROBI_HOME)/lib $(OMP_RPATH)

BUILD  := build
SHLIB  := libdw.$(SHLIB_EXT)
TARGET := tests/test_dw

.PHONY: all lib clean run

all: $(SHLIB) $(TARGET)
lib: $(SHLIB)

$(SHLIB): $(BUILD)/dw.o
	$(CC) $(CFLAGS) $(LDFLAGS) $(SHLIB_FLAGS) -o $@ $^ $(LIBS)

$(TARGET): $(BUILD)/dw.o $(BUILD)/test_dw.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

$(BUILD)/dw.o: src/dw.c src/dw.h | $(BUILD)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(BUILD)/test_dw.o: tests/test_dw.c src/dw.h | $(BUILD)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(BUILD):
	mkdir -p $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD) $(SHLIB) $(TARGET)
