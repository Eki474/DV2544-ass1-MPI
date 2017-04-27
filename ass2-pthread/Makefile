SRCS=$(wildcard *.c)
BINS=$(SRCS:%.c=%)

CFLAGS=-Wall -g -O2 -std=c99 -pthread

all: $(BINS) 

$(BINS): %: %.c
	gcc -o $@ $(CFLAGS) $<

clean:
	rm -f $(BINS)
