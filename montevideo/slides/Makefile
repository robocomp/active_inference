DAYS := $(filter-out %_efe.tex,$(wildcard day*.tex))
PDFS := $(DAYS:.tex=.pdf)

all: $(PDFS)

figures/.day1: figures/make_day1.py
	python3 figures/make_day1.py && touch $@

day1.pdf: figures/.day1

%.pdf: %.tex preamble.tex
	pdflatex -interaction=nonstopmode -halt-on-error $< > /dev/null
	pdflatex -interaction=nonstopmode -halt-on-error $< > /dev/null
	@grep -E "undefined|Overfull .hbox .[0-9]{2,}" $*.log || true

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

.PHONY: all clean
