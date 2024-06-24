# Immune response and severity of COVID-19
This repository contains the code and datasets used to reproduce the results presented in the paper '_Investigating the relationship between the immune response and the severity of COVID-19: a large-cohort retrospective study_'. The preprint is available on [medRxiv](https://www.medrxiv.org/content/10.1101/2024.06.20.24309246v1).

### Requirements
* pandas version: 1.1.5
* numpy version: 1.19.5
* scipy version: 1.5.4
* statsmodels version: 0.12.2
* scikit-learn version: 0.24.1
* matplotlib version: 3.3.4

### Setting up parameters
Key parameters for the analysis are defined in the _Modules/Parameters.py_ file. The parameters to set for reproducing the results are:

* _age_min_: minimum patient age.
* _age_max_: maximum patient age.
* _donset_min_: minimum days between the onset of symptoms and hospitalization.
* _donset_max_: maximum days between the onset of symptoms and hospitalization.

Our study used the following parameter setups:

* Default Setup:
  _age_min=30_, _age_max=100_, _donset_min=0_, _donset_max=30_
* Setup 2:
  _age_min=30_, _age_max=70_, _donset_min=0_, _donset_max=30_
* Setup 3:
  _age_min=70_, _age_max=100_, _donset_min=0_, _donset_max=30_
* Setup 4:
  _age_min=30_, _age_max=100_, _donset_min=0_, _donset_max=10_
* Setup 5:
  _age_min=30_, _age_max=100_, _donset_min=11_, _donset_max=30_

### Datasets
The anonymized datasets required for the analysis can be found in the _Data_ directory.

### Running the analysis
To reproduce and visualize the results, use the Jupyter notebooks located in the _Notebooks_ directory:

Univariate logistic regression analysis:

* Notebook: _UnivariateLogisticRegression.ipynb_
* Description: this notebook evaluates univariate logistic regression models for each variable in _allinput_set_ specified in _Parameters.py_. Our analysis covers only the default parameter setup described above. The results are saved in the _Results_ directory.

Multivariate logistic regression analysis:

* Notebook: _MultivariateLogisticRegression.ipynb_
* Description: this notebook evaluates multivariate logistic regression models using the _FC_set_, _Dem_set_, _CK_set_, and _BM_set_ variables specified in _Parameters.py_. Our analysis covers all the parameter setups described above. The results are saved in the _Results_ directory.

Results visualization:

* Notebook: _ResultsVisualization.ipynb_
* Description: this notebook provides visualizations and interprets the results saved in the _Results_ directory.

Descriptive statistics analysis:

* Notebook: _DescriptiveStatistics.ipynb_
* Description: this notebook reproduces the descriptive statistical analysis from the paper. Note: It must be run with the default setup and after executing _UnivariateLogisticRegression.ipynb_, as it relies on the sorted AUC values of the _allinput_set_ produced by that notebook.

### Notes
Execution order: always start with _UnivariateLogisticRegression.ipynb_ to generate necessary outputs for subsequent notebooks.
