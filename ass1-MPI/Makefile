SRCS=$(wildcard *.c)
BINS=$(SRCS:%.c=%)

CFLAGS=-Wall

all: $(BINS) 

$(BINS): %: %.c
	mpicc -o $@ $(CFLAGS) $<

clean:
	rm -f $(BINS)
