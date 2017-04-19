SRCS=$(wildcard *.c)
BINS=$(SRCS:%.c=%)

MPICC=mpicc
CFLAGS=-Wall -g -O3

all: $(BINS) 

$(BINS): %: %.c
	$(MPICC) -o $@ $(CFLAGS) $<

clean:
	rm -f $(BINS)
