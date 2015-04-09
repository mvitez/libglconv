UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)
LBITS := $(shell getconf LONG_BIT)

INCLUDE = -I. -I/usr/local/include -I$(HOME)/torch/install/include
LDFLAGS := -lm -fopenmp
LIBOPTS = -shared -L/usr/local/lib/lua/5.1 -L/usr/local/lib -L$(HOME)/torch/install/lib -L$(HOME)/torch/install/lib/lua/5.1
CFLAGS = -O3 -fopenmp -c -fpic -Wall -DUSEOMP
FILES = libglconv.o
LIBFILES = libglconv.so
CC = gcc

ifneq ($(filter arm%,$(UNAME_P)),)
	CFLAGS += -mfpu=neon
	LDFLAGS += -lGLESv2 -lEGL
else
	CFLAGS += -DUSEGLX
	LDFLAGS += -L/usr/lib/nvidia-331 -lGL -lX11
endif

ifeq ($(UNAME_S),Darwin)
	LDFLAGS += -lTH -lluajit -lluaT
endif

.PHONY : all clean install uninstall
all : $(LIBFILES) glconv

.c.o:
	$(CC) $(CFLAGS) $(INCLUDE) $<

glconv.o: libglconv.c
	$(CC) $(CFLAGS) $(INCLUDE)  -DNOLIB libglconv.c -o glconv.o

libglconv.so : $(FILES)
	$(CC) $(FILES) $(LIBOPTS) -o $@ $(LDFLAGS)
	
glconv : glconv.o
	$(CC) glconv.o -o glconv $(LDFLAGS)

install : $(LIBFILES)
	sudo cp $(LIBFILES) /usr/local/lib/lua/5.1/

uninstall :
	sudo rm /usr/local/lib/lua/5.1/libglconv.so

clean :
	rm -f *.o $(LIBFILES) glconv
