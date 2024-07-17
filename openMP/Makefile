#
#
#

OMP=SSPP_s394369_openMP.o wtime.o mmio.o mms.o

LFLAGS=-fopenmp
CFLAGS=-O4 -fopenmp 


all: wtime.o mmio.o mms.o openMP

openMP: $(OMP)
	$(CC) $(LFLAGS) -o$@  $(OMP)

 
clean:
	/bin/rm -fr *.o


.c.o:
	$(CC) $(CFLAGS) -c -o$@ $< 
