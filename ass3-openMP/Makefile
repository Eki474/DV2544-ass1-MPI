SRCS=$(wildcard *.c)
BINS=$(SRCS:%.c=%)

CFLAGS=-Wall -g -O2 -std=c99 -fopenmp

all: $(BINS) 

$(BINS): %: %.c
	gcc -o $@ $(CFLAGS) $<

clean:
	rm -f $(BINS)
