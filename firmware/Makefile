# Host build for firmware/wisense/.
#
# csi_dsp.c has no ESP-IDF dependency, so it compiles and runs with a plain C
# compiler on a laptop. firmware/esp32s3_har builds the same file unmodified.
#
#   mingw32-make -C firmware test     (Windows, Strawberry/MinGW)
#   make -C firmware test             (Linux/macOS)

ifeq ($(origin CC),default)
CC := gcc
endif

CFLAGS  ?= -std=c11 -O1 -g \
           -Wall -Wextra -Werror \
           -Wconversion -Wsign-conversion -Wshadow -Wpointer-arith \
           -Wstrict-prototypes -Wmissing-prototypes
LDLIBS  ?= -lm

SRC     := wisense/csi_dsp.c wisense/csi_frame.c
BUILD   := build

ifeq ($(OS),Windows_NT)
EXE := .exe
else
EXE :=
endif

TESTS   := test_csi_dsp test_csi_frame
BINS    := $(addprefix $(BUILD)/,$(addsuffix $(EXE),$(TESTS)))

.PHONY: test clean

test: $(BINS)
	@echo "--- firmware host tests -------------------------------------"
	@fail=0; for t in $(BINS); do ./$$t || fail=1; done; \
	 echo "-------------------------------------------------------------"; \
	 if [ $$fail -eq 0 ]; then echo "all firmware tests passed"; \
	 else echo "FIRMWARE TESTS FAILED"; fi; exit $$fail

$(BUILD)/%$(EXE): tests/%.c $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $< $(SRC) $(LDLIBS)

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
