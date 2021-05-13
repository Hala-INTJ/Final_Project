# Final Project: Final Effluent Total Phosphorus Analysis

## Overview 
This project will work with data provided by a Wastewater Treatment Facility. The plant supervisor has to accommodate new limits coming into effect in the near future for Final Effluent Total Phosphorus (TP). There are strict limits on effluent released into the environment, and the goal is to determine if the TP can be predicted, or if an exceedance event can be classified from the existing plant SCADA data (flow rates, chemical dosing, lab analysis results, ...). The plant supervisor has some working theories on existing metrics which may contribute to TP exceedances such as: sludge buildup in primaries causing solids to carry over out of the primaries. This information will be used in building and training of one or more models. The goal is to create a 'warning system' to provide early notification when a TP exceedance may be imminent.
## Project Plan
### Week 1: Data Acquisition and Domain Knowledge Gathering
This week will focus on identifying data to collect, and a format for the data transfer. Meetings are planned with the plant supervisor to review the data provided, along with a plant schematic to fully-understand known relationships between various factors. For example, there may be parallel processes, and specific amounts of time between process areas, which are necessary to understand to prepare the data effectively. The final report will include a labelled diagram with key elements of the process.
### Week 2: Database Design, Implementation and ETL
Once the data has been collected and reviewed, a database design will be created and implemented. The data will then be transformed and loaded into the database, possibly with some summarization optimizations. The database design will be documented, as future data for use in predictions will be continuously added.
### Week 3: Build, Train and Evaluate Supervised Machine Learning Model(s) 
This project has a clear target which is Total Phosphorus. A variety of supervised machine learning models including Neural Networks will be evaluated to determine which are best suited to the provided dataset. It is expected that the data preprocessing will be similar for all models, and may include binning, normalization, standardization, and PCA. The data will be split into a training and testing set, and will be used for all models to determine comparative results. The final model(s) chosen will be documented, with an explanation of the model's confusion matrix and accuracy.
### Week 4: Final Report with Visualizations and Recommendations
The final results will be presented as a web page with the results of the models, and a set of visualizations to support the findings. If no model is able to achieve a 90% accuracy, recommendations on possible steps which could be taken to improve the results will be included.

## Project Deliverables

- Week 1: 
    - Initial meeting with the plant supervisor
    - 13 Excel spreadsheets were collected
    - The plant supervisor provided the following diagram for the process:
    ![](https://github.com/Hala-INTJ/Final_Project/blob/main/Resources/Process.png)

