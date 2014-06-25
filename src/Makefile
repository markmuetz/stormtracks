CC      = gcc
CFLAGS  = -fPIC -pedantic -Wall
LDFLAGS = -shared

../lib/libcvort.so: cvort.c
	$(CC) $(LDFLAGS) $(CFLAGS) $^ -o $@

clean:
	rm ../lib/*.so
