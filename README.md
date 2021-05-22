<style>
table th:first-of-type {
    width: 4%;
}
table th:nth-of-type(2) {
    width: 32%;
}
table th:nth-of-type(3) {
    width: 32%;
}
table th:nth-of-type(4) {
    width: 32%;
}
table td{
    vertical-align: top;
}
</style>
# Final Project: Final Effluent Total Phosphorus Analysis
## Overview 
This project will work with data provided by a Wastewater Treatment Facility. The plant supervisor has to accommodate new limits coming into effect in the near future for Final Effluent Total Phosphorus (TP). There are strict limits on effluent released into the environment, and the goal is to determine if the TP can be predicted, or if an exceedance event can be classified from the existing plant SCADA data (Supervisory Control and Data Acquisition) such as flow rates, chemical dosing and lab analysis results. The plant supervisor has some working theories on existing metrics which may contribute to TP exceedances such as: sludge buildup in primaries causing solids to carry over out of the primaries. This information will be used in building and training of one or more models. The goal is to create a 'warning system' to provide early notification when a TP exceedance may be imminent. The approach taken in this project is inspired by the research conducted by Emile Cornelissen in this paper "https://fse.studenttheses.ub.rug.nl/18915/1/Thesis_EmileCornelissen.pdf".
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
### Week 1: 
- Initial meeting with the plant supervisor
- 25 Excel spreadsheets were collected
- Preliminary review of data in excel spreadsheets
- Prepared an overview document with clarification questions for the plant supervisor [Machine Learning Documents Summary.xlsx](https://github.com/Hala-INTJ/Final_Project/blob/main/Resources/Machine%20Learning%20Documents%20Summary.xlsx)
- The plant supervisor provided the following diagram for the process:
![](https://github.com/Hala-INTJ/Final_Project/blob/main/Resources/Process.png)

### Week 2:
#### Progress review meeting with the plant supervisor
- The process flows through the plant are divided into 3 stages, which represent 6 independent trains
- Models will only be built for Stage 3, representing the newer portion of the plant
- Stage 3 is further subdivided into 6 independent sub-trains. Models will be built for each of these sub-trains with the target to be "Total Phosphorous" at the end of each sub-train.
    
#### Mapping tags to Trains
Updated the overview spreadsheet [Machine Learning Documents Summary.xlsx](https://github.com/Hala-INTJ/Final_Project/blob/main/Resources/Machine%20Learning%20Documents%20Summary.xlsx) to include the mapping of each tag into trains, stages and process areas. This was necessary for understanding the relationships between the tags and processes, and to ensure the dataset is complete. Each train is an independent path wastewater flows through the plant.

#### Database Design
For all the options below, there will be a table identifying all of the tags and other meta data relating to the collection of source files. Four options were identified and evaluated, and option 4 was selected. 

| Option | Description | Pros | Cons |
| :---: | :--- | :--- | :--- |
| 1 | Tables: 2<br />1 row per time/tag value (~435,000 rows) | * Add data for new tags without changing the schema | * Difficult to query to build pandas DataFrames |
| 2 | Tables: 2<br />1 column per tag (291 columns, ~1,500 rows) | * Easy to build pandas DataFrames | * Schema change every time adding a new tag |
| 3 | Tables: 26<br />1 table per excel file | * Convenient for loading data into the Database | * Will require several DataFrames merges to group data appropriately |
| 4 | Tables: 8<br />1 table per process area | * Good balance between single table and 1 table per excel file<br />* Allows clean DataFrames for data preparation and combinations<br />* Will require fewer DataFrames merges when preparing for ML models | * Some excel files will populate more than 1 table<br />* Some data may be replicated in more than 1 table |

Here is the link to the [ERD](https://github.com/Hala-INTJ/Final_Project/blob/main/Resources/ERD.png). The Tag table is used during the ETL process to translate the source tag names to ML tags, which are better suited to prepare data for machine learning models. The remaining tables represent process areas and are columnar, indexed by time for efficient retrieval for modeling.

#### ETL Implementation    
The data provided are daily time series data recorded at a wastewater treatment plant (WWTP). There were many challenges posed by the quality and consistency of the data:
            
- there were text values (eg. NT, OS, ND)
- some values were prefixed with '<' or '>'
- some columns had very sparse data (eg. some analytes)
- unexpected negative values
- complex and long column headers 
        
The ETL process was implemented using Jupyter notebooks, and with common, reusable functions collected in a separate Python script [ETL.py](https://github.com/Hala-INTJ/Final_Project/blob/main/ETL/ETL.py). A Jupyter notebook was created for each Excel file, whare are found in [ETL Folder](https://github.com/Hala-INTJ/Final_Project/tree/main/ETL). The process for each file involved the following steps:
- Load the data into a DataFrame from Excel
- Remove Text and replace with NaN
- Remove less than sign, and replace with 1/2 the original value (<x becomes x/2)
- Remove greater than sign, and replace with the original value (>x becomes x)
- Remove 'Time' column
- Convert the data type to numeric for all columns except the 'Time' column
- Identify outliers using STL (Seasonal and Trend decomposition using Loess)
- Remove outliers and replace with NaN
- Replace negative values with 0
- To replace NaN values, interpolate using 'pchip' method (Piecewise Cubic Hermite Interpolating Polynomial)
- Re-insert the 'Time' column
- Converted column headers to ML Tags using a standardized naming convention (Train, Stage, Process Area, Pri/Sec, Type)
- Write the DataFrame to Postgres Database

The ETL implementations include a number of visualizations before and after interpolation, which were used to inspect and review the data. 

The ETL process resulted in the following tables:

| Table Name | Number of Records | Number of Columns |
| --- | --- | --- |
| Tags | 290 | 11 |
| Influent | 1592 | 15 |
| Preliminary | 1592 | 5 |
| Primary | 1592 | 111 |
| Aeration | 1592 | 54 |
| Secondary | 1592 | 97 |
| Incineration | 1591 | 2 |
| Effluent | 1592| 13 |
## Technical Terms

**Aeration**: The process of adding air to water. In wastewater treatment, air is added to refreshen wastewater and to keep solids in suspension. With mixtures of wastewater and activated sludge, adding air provides mixing and oxygen for the microorganisms treating the wastewater. 

**BOD**: Biochemical Oxygen on Demand -- an indirect reading of the organic content present in wastewater. Specifically, it refers to the amount of oxygen consumed to biologically degrade the organic material. It’s very expensive to treat, typically requiring a biological treatment technology like activated sludge.

**cBOD**: Carbonaceous Biochemical Oxygen Demand -- the amount of oxygen required, under controlled conditions, to oxidize any carbon containing matter present in a water by biological means.

**COD**: Chemical Oxygen Demand -- an indirect reading of the organic content of wastewater. Specifically, it refers to the amount of oxygen that is required to chemically degrade the organic material.

**DO**: Dissolved Oxygen -- an indication of how much oxygen is present in water. 

**Effluent**: Wastewater (treated or untreated) that flows out of a facility.

**HW**: Headworks -- the facilities where wastewater enters a wastewater treatment plant.

**Influent**: Wastewater received at a wastewater treatment facility and includes waste from homes, businesses and industry; a mixture of water and dissolved and suspended solids.

**MLSS**: Mixed Liquor Suspended Solids -- the total concentration of solids in the aeration tanks and includes both inorganic and organic solids.

**Sludge**: A mixture of solids and water produced during the treatment of wastewater.

**SRP**: Soluble Reactive Phosphorous, or Dissolved Phosphorous -- a measure of orthophosphate, the filterable (soluble, inorganic) fraction of phosphorus.

**TKN**: Total Kjeldahl Nitrogen -- a pollutant found in domestic sewage that is typically a surcharge parameter for industries.

**TP**: Total phosphorus -- a measure of all the forms of phosphorus, dissolved or particulate.

**TSS**: Total Suspended Solids -- solids in water that can be trapped by a filter.

**WAS**: Waste Activated Sludge -- the activated sludge removed from the secondary treatment process. 















