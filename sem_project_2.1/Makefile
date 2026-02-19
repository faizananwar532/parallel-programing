CC = mpicc
CFLAGS = -O3 -Wall
TARGET = matmul

all: $(TARGET)

$(TARGET): matmul.c
	$(CC) $(CFLAGS) -o $(TARGET) matmul.c -lm

clean:
	rm -f $(TARGET)

.PHONY: all clean
