SRCS=$(wildcard *.c)
BINS=$(SRCS:%.c=%)

MPICC=/opt/mpich/bin/mpicc
CFLAGS=-Wall -g -O3

all: $(BINS) 

$(BINS): %: %.c
	$(MPICC) -o $@ $(CFLAGS) $<

clean:
	rm -f $(BINS)
