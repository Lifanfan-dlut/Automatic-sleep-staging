# Automatic-sleep-staging
This project includes a class-balance preprocessing method for automatic sleep staging in OSA .

## Introduction
Data used in this study can be download in  https://sleepdata.org/datasets/shhs/.

## Usage
Then, follow these steps and you will get automatic sleep staging model.

1. **Run preprocessong_edf_xml.py**. This file convert PSG files to epoch-by-epoch EEG segments.
2. **Run Training_model.py**. This file includes the class-balance trainng method in **batch_data_generate**.
3. **Run Test_model.py**. This file pis used to test model.

## Author
*LI Fanfan, 3rd Ph.D in School of Biomedical Engineering of Dalian University of Technology.

## Dependencies
* pyedflib
* numpy,os, re
* GPU
* keras, backend tensorflow
