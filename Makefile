CC = mpicc
CFLAGS = -O2 -Wall -std=c99
LDFLAGS = -lm

SRCDIR = .
OBJDIR = obj
BINDIR = bin

SRCS := main.c $(wildcard tasks/*.c)
OBJECTS := $(patsubst %.c,$(OBJDIR)/%.o,$(SRCS))

TARGET_PI = $(BINDIR)/pi_calculator
TARGET_MATRIX = $(BINDIR)/matrix_multiplier
TARGET_MAIN = $(BINDIR)/main

all: $(TARGET_PI) $(TARGET_MATRIX) $(TARGET_MAIN)

$(TARGET_PI): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_MATRIX): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_MAIN): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

clean-results:
	rm -rf results

distclean: clean clean-results

run_pi: $(TARGET_PI)
	@mkdir -p results
	mpirun -np 2 $(TARGET_PI) 1

run_matrix: $(TARGET_MATRIX)
	@mkdir -p results
	mpirun -np 2 $(TARGET_MATRIX) 2

.PHONY: all clean clean-results distclean run_pi run_matrix