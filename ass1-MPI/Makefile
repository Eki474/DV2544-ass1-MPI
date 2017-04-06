SRCS=$(wildcard *.c)
BINS=$(SRCS:%.c=%)

MPICC=/opt/mpich/bin/mpicc
CFLAGS=-Wall -g

all: $(BINS) 

$(BINS): %: %.c
	$(MPICC) -o $@ $(CFLAGS) $<

clean:
	rm -f $(BINS)
