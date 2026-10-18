LATEXMK ?= latexmk
LECTURE ?= lecture00

.PHONY: all clean

all:
	cd "$(LECTURE)" && $(LATEXMK) main.tex

clean:
	cd "$(LECTURE)" && $(LATEXMK) -c main.tex
