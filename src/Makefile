CC      = gcc
CFLAGS  = -fPIC -pedantic -Wall
LDFLAGS = -shared

libcvort.so: cvort.c
	$(CC) $(LDFLAGS) $(CFLAGS) $^ -o $@

clean:
	rm *.so
