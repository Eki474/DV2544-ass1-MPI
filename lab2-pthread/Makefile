SRCS=$(wildcard *.c)
BINS=$(SRCS:%.c=%)

CFLAGS=-Wall -g -O2 -pthread

all: $(BINS) 

$(BINS): %: %.c
	gcc -o $@ $(CFLAGS) $<

clean:
	rm -f $(BINS)
