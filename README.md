# Rosa Biotech
This repository is for Rosa code used in various publications.

The fluorescence data analysis to machine learning pipeline is the same across all publications. This involves the transformation of fluorescence data in plate reader format into a dataframe with metadata for analysis attached. The code then performs normalisation on this data in the form of min max scaling, bespoke to the plate design. Various visualisations of this data is then produced. The data is then prepared for statistical modelling/machine learning. A pipeline compares a selection of models and evaluates the optimum performance for a dataset.
Publication specific code (e.g. running docking simulations, web scraping from the HMDB database and metabolomics analysis) is stored on publication specific branches.

This code is slightly altered from what was originally run in Rosa, due to the fact this code is being made publically accessible. The original Rosa code makes use of a cloud database. The data is transformed using SQL to get it into the dataframe format. Due to the fact the database is private, and also will be archived, the already processed data is included instead in this repository, and code used to make connections to this database is not included in this repository.
