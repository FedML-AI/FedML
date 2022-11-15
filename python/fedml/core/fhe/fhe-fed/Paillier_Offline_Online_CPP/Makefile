CXX = g++
CC := gcc # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity

SRCDIR := src
BUILDDIR := build
TARGET := bin/main

MKDIR = mkdir -p $(BUILDDIR)
 
CSRCEXT := c
CXXSRCEXT := cpp

CSOURCES := $(shell find $(SRCDIR) -type f -name *.$(CSRCEXT))
CXXSOURCES := $(shell find $(SRCDIR) -type f -name *.$(CXXSRCEXT))


OBJECTSC := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CSOURCES:.$(CSRCEXT)=.o))
OBJECTSCXX := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CXXSOURCES:.$(CXXSRCEXT)=.o))


CFLAGS := -fopenmp -g2 -O3
CXXFLAGS := -fopenmp -g2 -O3 -std=c++11 -lgmp -lgmpxx -lcryptopp

LIB := -L /usr/local/lib 
INC := -I /usr/local/include


all: server

server: 
	$(MKDIR)
	$(CC) $(CSOURCES) $(CFLAGS) -c $(INC) -o $(OBJECTSC)
	$(CXX) $(OBJECTSC) $(CXXSOURCES) $(CXXFLAGS) $(INC) $(LIB) -o $(TARGET)

.PHONY: clean
clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)