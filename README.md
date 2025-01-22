# **Optimizing Metagenome Analysis for Early Detection of Colorectal Cancer: Benchmarking Bioinformatics Approaches and Advancing Cross-Cohort Prediction**
Colorectal cancer (CRC) continues to be a major global public health challenge. In this study, we conducted a systematic analysis of over 2,000 gut metagenomic samples from 15 globally-sampled public and in-house cohorts. By benchmarking 1,080 analytical combinations across multiple analytical steps, we established a bioinformatics workflow for metagenome-based CRC detection. However, early-stage prediction of CRC, particularly at the precancerous adenomas (ADA) stage, remains challenging due to the instability of microbial signatures across cohorts. To address this, we developed an instance-based transfer learning strategy, Meta-iTL, which improved ADA detection in a newly recruited cohort. This study not only provides a comprehensive bioinformatics guild for metagenomic data processing and modeling but also advances the development and application of non-invasive approaches for the early screening and prevention of CRC.
## Table of Contents
* [Installation](#installation)
  * [Software](#software)
  * [Software setup](#software-setup)
* [CRC Benchmark Workflow](#CRC-Benchmark-Workflow)
  * [Stage 1 Differential signature identification](#stage-1-differential-signature-identification)
  * [Stage 2 Meta analysis](#stage-2-meta-analysis)
  * [Stage 3 Features selection](#stage-3-features-selection)
  * [Stage 4 Machine learning models](#stage-4-machine-learning-models)
* [Meta_iTL](#Meta_iTL)
  * [Subset selection and matching by a mutual nearest neighbor set algorithm](#Subset-selection-and-matching-by-a-mutual-nearest-neighbor-set-algorithm)
  * [Feature extraction and initial model](#Feature-extraction-and-initial-model)
  * [Final model training](#Final-model-training)
## Installation
### Software
- R v.4.1.3 or newer (https://www.r-project.org)
- Python3 v3.9 or newer (https://www.python.org)
#### R packages
- BiocManager (https://github.com/Bioconductor/BiocManager) to ensure the installation of the following packages and their dependencies.
- MMUPHin (https://huttenhower.sph.harvard.edu/mmuphin)
- MaAsLin2 (https://github.com/biobakery/Maaslin2)
- dplyr (https://dplyr.tidyverse.org/)
- vegan (https://github.com/vegandevs/vegan)
- labdsv (https://github.com/cran/labdsv)
- microeco (https://github.com/ChiLiubio/microeco)
#### python packages
- pandas (https://pandas.pydata.org)
- NumPy (https://numpy.org/)
- scikit-learn (https://scikit-learn.org)
- bioinfokit (https://github.com/reneshbedre/bioinfokit)
- Matplotlib (https://matplotlib.org/)
- seaborn (https://seaborn.pydata.org/)
- GeoPandas(https://geopandas.org/)
### Software setup
#### Installation of R and R packages
Installation of R on different platforms can be conducted following the instructions on the official website (https://www.r-project.org/). All R packages used in this protocol can be installed following given commands.
```R
> install.packages(Package_Name)
```
or
```R
> if (!requireNamespace(“BiocManager”, quietly = TRUE)) 
> install.packages(“BiocManager”)
> BiocManager::install(Package_Name)
```
#### Installation of python and python packages
Python can be downloaded and installed from its official website (https://www.python.org/), and all python packages could be installed using pip.
```python
 pip install Package_Name
```
## CRC Benchmark Workflow
### Stage 1 Preprocessing of microbial features 
Rare signatures, those with low occurrence rates across cohorts are discarded  to ensure that identified *features* are reproducible and could be applied to prospective cohorts. 
```R
  Rscript ./Benchmark/Process_profile/main_process.R
```
### Stage 2 Meta analysis 
In meta-analysis, heterogeneity among cohorts caused by differences in confounding factors is inevitable, which can seriously affect the identification of differential features downstream. Permutational multivariate analysis of variance (PERMANOVA) test is one of the most widely used nonparametric methods for fitting multivariate models based on differential measures in microbial research, which quantifies the microbial variation attributed to each metadata variable, thereby assigning representatives to evaluate confounding effects. The PERMANOVA test here is based on Bray-Curtis. For each metadata variable, the coefficient of determination *F* value and *p* value are calculated to explain how the variation is attributed. The variables with the greatest impact on the microbial profile are considered as the main batch. Principal coordinate analysis (PCoA) plots are also provided.
```R
  Rscript ./Benchmark/Figers/Fig2.R
```
The script will compute the variance explained by the disease status (i.e. CRC) and by potential confounding variables (e.g. Age, Sex, BMI, etc.).  The script will also produce a plot contrasting the variance explained by the potential confounder and by disease status.
### Stage 3 Features selection   
Filter features based on 8 common feature selection tools.
```R
  Rscript ./Benchmark/Process_profile/2_select_feature.R
```
- The script **2_select_feature.R** includes feature selection methods such as LEfSe, MaAsLin2, ANCOM-II, T-test, metagenomeSeq, and the two-sided Wilcoxon test.
```python
  python ./Benchmark/Process_profile/2_select_feature.py
```
- The script **2_select_feature.py** incorporates **RFECV** for feature selection.
```R
  Rscript ./Benchmark/Process_profile/3_merger_feature.R
```
- The script **3_merger_feature** consolidates the features selected by different tools, including LEfSe, MaAsLin2, ANCOM-II, T-test, metagenomeSeq, and the two-sided Wilcoxon test, into a single table. Each column in the table represents the features selected by a specific tool.
```python
  python ./Benchmark/Process_profile/4_feature_frequence.py
```
- The script **4_feature_frequence.py** calculates the frequency at which each feature is selected by the different tools.
```python
  python ./Benchmark/Process_profile/5_Synergistic_feature.py
```
- The script **5_Synergistic_feature.py** is a synergistic feature selection tool that consolidates features identified by LEfSe, MaAsLin2, and ANCOM-II. Then this set was refined by the Max-Relevance and Min-Redundancy (mRMR) method alongside an iterative feature elimination (IFE) step to finalize the optimal biomarkers for CRC.
### Stage 4 Machine learning models
#### *Model construction* 
This step provides optional classifier selection for subsequent steps where the performances of every ML algorithm are generally assessed using all differential signatures. The output file contains the cross-validation AUC, AUPR, MCC, specificity, sensitivity, accuracy, precision, and F1 score of all classification models built with these various algorithms. 
```
  python ./Benchmark/benchmark.py
```
- The script **2_select_feature.R** includes machine learning methods such as (RF, XGBoost, MLP, KNN, and SVM).
#### *Model evaluation* 
As stated above, this step provides extensive internal validations to ensure the robustness and reproducibility of identified *candidate features* in different cohorts via intra-cohort validation, cohort-to-cohort transfer, and LOCO validation.
```
  python ./Benchmark/Cross_cohort_valid.py
```
As the best-performing *candidate features* and classification model are established, the test dataset is used to externally validate their generalizability. The input external metadata and microbial relative profiles need to be in the same format as initial input files for the training dataset. This step returns the overall performance of the model and its AUC plot.  
```
  python ./Benchmark/ExternalValid.py
```
-  The script **Cross_cohort_valid.py** achieved multi-cohort cross-validation, inter-cohort validation and leave-one-out-of-the-dataset (LODO) validation, effectively mitigating the effects of technical and geographical variations.
-  The script **ExternalValid.py**  achieved two independent cohorts(CHN_SH-2 and CHN_SH-3).
## Meta_iTL
 We developed Meta-iTL, an instance-based transfer learning (TL) modeling strategy. By leveraging knowledge transfer between samples to be tested (i.e., target domain) and existing data repository (i.e., source domain), Meta-iTL overcomes challenges limited by the cohort-specific effects and scarcity of ADA cases for training, improving cross-cohort applicability of data models. In summary, our study provides a reproducible bioinformatics framework for cross-cohort CRC and ADA detection. This approach advances the development and application of non-invasive early screening, offering significant potential for improving early diagnosis and prevention of CRC.
####  *Subset selection and matching by a mutual nearest neighbor set algorithm*
```python
  python ./Meta_iTL/script/01_findMNN.py -t CHN_SH-CRC-4 -r S0.3
```
	-t: Specifies the name of the target study. 
	-r: Defines the percentage of target domain samples selected for transfer learning. 
-  The script **01_findMNN.py** identifies and forms the mutual nearest neighbor set.
#### *Feature extraction and initial model*
```
  Rscript ./Meta_iTL/script/02_findfeatures.R
```
-  The script **02_findfeatures.R** includes feature selection methods (LEfSe, MaAsLin2, ANCOM-II).
```
  python ./Meta_iTL/script/03_mergefeatures.py -t CHN_SH-CRC-4 -d Meta_iTL -r S0.3
```
```
-t: Specifies the name of the target study.  
-d: Indicates whether to use the original cohort or apply Meta-iTL for predictions in the target domain.  
-r: Defines the percentage of target domain samples selected for transfer learning.  
```
-  The script **03_mergefeatures.py** consolidates the features selected by different tools, including LEfSe, MaAsLin2 and ANCOM-II,  into a single table. Each column in the table represents the features selected by a specific tool.
```
  python ./Meta_iTL/script/04_feature_frequency.py -t CHN_SH-CRC-4 -d Meta_iTL -r S0.3
```
```
-t: Specifies the name of the target study.  
-d: Indicates whether to use the original cohort or apply Meta-iTL for predictions in the target domain.  
-r: Defines the percentage of target domain samples selected for transfer learning.  
```
-  The script **04_feature_frequency.py** calculates the frequency at which each feature is selected by the different tools.
```
  python ./Meta_iTL/script/04_optimalfeatures.py -t CHN_SH-CRC-4 -d Meta_iTL -r S0.3 -f 0  
```
```
-t: Specifies the name of the target study.  
-d: Indicates whether to use the original cohort or apply Meta-iTL for predictions in the target domain.  
-r: Defines the percentage of target domain samples selected for transfer learning.   
-f: Features with frequencies exceeding a specified threshold are selected for subsequent feature optimization(Choose based on actual situation).
```
-  The script **04_optimalfeatures.py** s a synergistic feature selection tool that consolidates features identified by LEfSe, MaAsLin2, and ANCOM-II. Then this set was refined by the Max-Relevance and Min-Redundancy (mRMR) method alongside an iterative feature elimination (IFE) step to finalize the optimal biomarkers for adenoma (ADA).
#### *Final model training*
```
  python ./Meta_iTL/script/05_Meta_iTL_Result.py -t CHN_SH-CRC-4 -d Meta_iTL -r S0.3
```
```
-t: Specifies the name of the target study.  
-d: Indicates whether to use the original cohort or apply Meta-iTL for predictions in the target domain.  
-r: Defines the percentage of target domain samples selected for transfer learning. 
```
-  The script **05_Meta_iTL_Result.py** predicts the target domain cohort using either the transfer learning model or the non-transfer learning model.