.PHONY: all clean watch
all:
	latexmk -pdf -interaction=nonstopmode main.tex
watch:
	latexmk -pdf -pvc -interaction=nonstopmode main.tex
clean:
	latexmk -C
	rm -f *.bbl *.blg *.spl
