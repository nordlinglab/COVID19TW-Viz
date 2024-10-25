# COVID19TW-Viz
Companion code for visualization of Taiwanese COVID-19 data presented in our publication in Scientific Data:
> Wu, Y. H., & Nordling, T. E. (2024). [A structured course of disease dataset with contact tracing information in Taiwan for COVID-19 modelling](https://doi.org/10.1038/s41597-024-03627-z). Scientific Data, 11(1), 821.

## Overview
This repository contains data and code to help researchers and analysts explore Taiwan's COVID-19 cases and their contact tracing information. The visualizations provide insights into course of disease patterns and contact tracing information during the pandemic.

## Data Files
- `taiwan_covid.xlsx`
  - Raw COVID-19 case data from Taiwan
  - Used as input for our data processing pipeline

- `figshare_taiwan_covid.xlsx`
  - Cleaned and structured dataset
  - Contains standardized case information and contact tracing data
  - Also available on [Figshare](https://doi.org/10.6084/m9.figshare.24623964.v2)

- `codebook.md`
  - Detailed documentation of all variables
  - Explains data headers and contact type classifications
  - Essential reference for understanding the dataset structure

## Code Files
- `plot_taiwan_data.ipynb`
  - Jupyter notebook containing visualization code
  - Reproduces all figures from our paper

- `rw_data_processing.py`
  - Core data processing functions
  - Handles data cleaning and transformation

- `rw_visualization.mplstyle`
  - Custom Matplotlib style configuration
  - Ensures consistent and publication-ready figures


## Citation
If you use this data or code in your research, please cite our paper:
```bibtex
@article{Wu2024Structured,
    abstract = {The COVID-19 pandemic has flooded open databases with population-level data. However, individual-level structured data, such as the course of disease and contact tracing information, is almost non-existent in open databases. Publish a structured and cleaned COVID-19 dataset with the course of disease and contact tracing information for easy benchmarking of COVID-19 models. We gathered data from Taiwanese open databases and daily news reports. The outcome is a structured quantitative dataset encompassing the course of the disease of Taiwanese individuals, alongside their contact tracing information. Our dataset comprises 579 confirmed cases covering the period from January 21, to November 9, 2020, when the original SARS-CoV-2 virus was most prevalent in Taiwan. The data include features such as travel history, age, gender, symptoms, contact types between cases, date of symptoms onset, confirmed, critically ill, recovered, and dead. We also include the daily summary data at population-level from January 21, 2020, to May 23, 2022. Our data can help enhance epidemiological modelling.},
    author = {Wu, Yu-Heng and Nordling, Torbj{\"{o}}rn E M},
    doi = {10.1038/s41597-024-03627-z},
    issn = {2052-4463},
    journal = {Scientific Data},
    month = {Jul},
    pages = {821},
    title = {{A structured course of disease dataset with contact tracing information in Taiwan for COVID-19 modelling}},
    url = {https://doi.org/10.1038/s41597-024-03627-z},
    volume = {11},
    year = {2024}
}