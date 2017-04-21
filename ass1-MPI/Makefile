SRCS=$(wildcard *.c)
BINS=$(SRCS:%.c=%)

MPICC=mpicc
CFLAGS=-Wall -g -O2

all: $(BINS) 

$(BINS): %: %.c
	$(MPICC) -o $@ $(CFLAGS) $<

clean:
	rm -f $(BINS)
