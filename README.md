# Parliament Speech Classifier

This project aims to train a classifier for predicting political affiliation based on speeches held in the Finnish Parliament. This project is part of the Semparl project: https://seco.cs.aalto.fi/projects/semparl/. The data used for training comes from Parliamenttisampo: https://parlamenttisampo.fi/fi/

The code is divided into three parts. First, you should run the read_xml.py script, to parse the raw xml files, pick the relevant data and convert into csv files. Then, run create_datasets.py to furher process the data and prepare it for the classifier. Finally, run classifier-shap.py, which trains the model and the calculates SHAP values. The SHAP part of the code is commented our because it runs into OOM errors. If you have the same problem, you can calculate SHAP values after training by running the file run_shap.py to run to get explanations for a single model or shap_stability_combined_tokens.py to get aggregate keywords for multiple models.
