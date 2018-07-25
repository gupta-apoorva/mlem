CC      = g++
RM      = rm -f

-include Makefile.config

CFLAGS  = -O3 -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__
LFLAGS  = -O3
IFLAGS = -lboost_system -lboost_filesystem -lboost_program_options -lSDL2main -lSDL2

SOURCES = csr4matrix.cpp scannerconfig.cpp symhandler.cpp timers.cpp vector3d.cpp
HEADERS = csr4matrix.hpp scannerconfig.hpp symhandler.hpp timers.hpp vector3d.hpp matrixelement.hpp
OBJECTS = $(SOURCES:%.cpp=%.o)

all: dispcsr4spy	

dispcsr4spy: dispcsr4spy.o $(OBJECTS)
	$(CC) $(CFLAGS) $(LFLAGS) -o $@ dispcsr4spy.o $(OBJECTS) $(IFLAGS)

dispcsr4spy.o: dispcsr4spy.cpp 
	$(CC) $(CFLAGS) $(LFLAGS) -I. -o $@ -c $< $(IFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(LFLAGS) -o $@ -c $< $(IFLAGS)

clean:
	- $(RM) *.o dispcsr4spy

distclean: clean
	- $(RM) *.c~ *.h~



