$pdf_mode = 1;
$out_dir = 'build';
$aux_dir = 'build';
$bibtex_use = 1;
$pdflatex = 'case "%S" in *main.tex) latexmk -pdf paper.tex || exit $?;; esac; pdflatex -interaction=nonstopmode -halt-on-error %O %S';
$clean_ext = 'acn acr alg aux bbl bcf blg fdb_latexmk fls glg glo gls idx ilg ind ist lof log lot out run.xml toc';
