CC      = gcc
CFLAGS  = -fPIC -pedantic -Wall -o3 -funroll-loops
LDFLAGS = -shared

../stormtracks.so: cvort.c cextrema.c
	$(CC) $(LDFLAGS) $(CFLAGS) $^ -o $@

clean:
	rm ../stormtracks.so
