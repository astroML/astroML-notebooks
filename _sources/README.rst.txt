AstroML notebooks
-----------------

This repository shares notebooks developed for AstroML. The material
reproduces many of the figures from the book `Statistics, Data Mining, and
Machine Learning in Astronomy`, and therefore organized in topics following
the chapters. We focused on the later chapters that
have actual applications using astronomical data.
Narrative commentary is provided, however some sections maybe less
self-explanatory without the book as a reference.
The table of contents of the book can be found on the main astroML webpage, `www.astroml.org`.


Contributing
^^^^^^^^^^^^

Contributions are welcome in the form of pull requests.

This repository uses Jupytext and MyST Markdown Notebook to generate static
html pages. We store both the linked ``.ipynb`` and ``.md`` files.


Smaller changes
"""""""""""""""

For making smaller changes to the text content of the notebooks, please edit
the .md files using either the GitHub interface or a text editor.


Editing ipynb
"""""""""""""

For larger changes to the code and outputs, the easiest approach is to edit
the notebooks directly.


Rendering the notebooks
"""""""""""""""""""""""

You will need to install some extra dependencies for rendering the notebooks to html pages, install them via:

``pip install -r doc-requirements.txt``

Running some of the notebooks takes significant time, therefore we store the outputs in them and therefore
building the pages without executing the notebooks:

``sphinx-build -b html -D jupyter_execute_notebooks=off . _build/html``

The rendered notebooks then available in ``_build/html/index.html``
