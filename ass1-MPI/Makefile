SRCS=$(wildcard *.c)
BINS=$(SRCS:%.c=%)

MPICC=mpicc
CFLAGS=-Wall -g -O2 -std=c99

all: $(BINS) 

$(BINS): %: %.c
	$(MPICC) -o $@ $(CFLAGS) $<

clean:
	rm -f $(BINS)
