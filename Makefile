TOP := $(shell pwd)
ARCH := $(shell getconf LONG_BIT)
UNAME_S := $(shell uname -s)
MKDIR_P := $(shell mkdir -p build)
MKDIR_TEST := $(shell mkdir -p build/test)
MKDIR_LIB := $(shell mkdir -p build/lib/ioc_container)

LIB_DIR=lib
IDIR=include
CC=g++

# CPP_FLAGS=-I. -W -Wall -Werror -pedantic -std=c++11 -O3 -D_FILE_OFFSET_BITS=64
CPP_FLAGS=-g -I. -W -Wall -pedantic -std=c++11 -D_FILE_OFFSET_BITS=64 -O0

CPP_FLAGS_32 := -Dx86 -m32
CPP_FLAGS_64 := -Dx64 -m64
CPP_FLAGS += $(CPP_FLAGS_$(ARCH))

BUILD_DIR=build

# ------------------------------
# Libs
# ------------------------------

# _DEPS = ioc_container.hpp
# DEPS = $(patsubst %,$(LIB_DIR)/ioc_container/%,$(_DEPS))

# _OBJ = 
# OBJ = $(patsubst %,$(BUILD_DIR)/lib/ioc_container/%,$(_OBJ))
# 
# INC += -I$(LIB_DIR)/ioc_container/
# 
# $(BUILD_DIR)/lib/ioc_container/%.o: $(LIB_DIR)/ioc_container/%.cpp $(DEPS)
# 	$(CC) -c -o $@ $< $(CPP_FLAGS) -I$(IDIR) $(INC)
# 
# $(BUILD_DIR)/lib/ioc_container/libioccontainer.a: $(OBJ)
# 	ar rcs $@ $^

# LDLIBS += $(BUILD_DIR)/$(LIB_DIR)/ioc_container/libioccontainer.a

# LDLIBS += -lprofiler

# ------------------------------
# Executables
# ------------------------------

_DEPS = bootstrapper.hpp neuron_trainer.hpp neural_network.hpp perceptron_trainer.hpp backpropagation_trainer.hpp ioc_container.hpp 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = perceptron_trainer.o backpropagation_trainer.o neural_net.o
OBJ = $(patsubst %,$(BUILD_DIR)/%,$(_OBJ))

$(BUILD_DIR)/%.o: src/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CPP_FLAGS) -I$(IDIR) $(INC)

$(BUILD_DIR)/main: $(BUILD_DIR)/main.o $(OBJ)
	$(CC) -o $@ $^ $(LDLIBS) $(CPP_FLAGS)

# Tests.

_TESTS = backpropagation_test # perceptron_test 
TESTS = $(patsubst %,$(BUILD_DIR)/test/%,$(_TESTS))

$(BUILD_DIR)/test/%.o: test/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CPP_FLAGS) -I$(IDIR) $(INC)

$(BUILD_DIR)/test/%: $(BUILD_DIR)/test/%.o $(OBJ)
	$(CC) -o $@ $^ $(LDLIBS) $(CPP_FLAGS)
# 

# all: $(BUILD_DIR)/lib/ioc_container/libioccontainer.a $(BUILD_DIR)/main 
all: $(BUILD_DIR)/main $(TESTS)


.PHONY: test

test:
	@for i in $(TESTS); do \
          $$i; \
          if [ $$? -eq 0 ]; then \
            echo "$$i (PASSED)"; else echo "$$i (FAILED)"; fi; \
        done;               

# Tasks.

.PHONY: clean

clean:
	rm -rf $(BUILD_DIR)
