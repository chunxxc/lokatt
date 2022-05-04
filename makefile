OUT = bin/main
CC = nvcc
INC = -I/usr/include/hdf5/serial
LIBS = -lhdf5_serial -lhdf5_cpp -lboost_system -lboost_filesystem -lboost_program_options
CFLAGS = -Xcompiler -O3 -Xptxas -O3 -maxrregcount 32 -use_fast_math

OBJ = main.o dnaseq_beam_makefiletest.o

main :
	nvcc -c -o ./tensorflow_op/dnaseq_beam.o ./tensorflow_op/dnaseq_beam.cu $(CFLAGS)
	nvcc $(INC) -c -o main.o main.cpp $(CFLAGS)
	nvcc main.o dnaseq_beam_makefiletest.o $(LIBS) $(CFLAGS)
	./tensorflow_op/a.out
