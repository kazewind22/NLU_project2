output/%.pdf: %.tex
	mkdir -p output
	pdflatex -output-directory=output -shell-escape $<
	bibtex output/$*
	pdflatex -output-directory=output -shell-escape $<
	pdflatex -output-directory=output -shell-escape $<

all: output/report.pdf

clean:
	rm -rf output
