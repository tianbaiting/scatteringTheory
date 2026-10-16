TEX = xelatex
LECTURE = lecture01
all:
	cd $(LECTURE) && $(TEX) -interaction=nonstopmode -halt-on-error main.tex
	cd $(LECTURE) && $(TEX) -interaction=nonstopmode -halt-on-error main.tex
clean:
	cd $(LECTURE) && rm -f *.aux *.log *.nav *.out *.snm *.toc *.synctex.gz
