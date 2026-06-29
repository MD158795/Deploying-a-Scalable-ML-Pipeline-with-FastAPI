# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a supervised binary classification model trained to predit whether an individual's income would be greater than 50k, or less than or equal to 50k. The model uses a RandomForestClassifier from scikit-learn. Categorical features are processed using a one-hot encoding method. The target label was processed using a labal binarizer. 
## Intended Use
This model is intended to be used as a demonstration of end to end machine learning for the Udacity MLDevOps nanodegree. It is for educational purposes only. 
## Training Data
The model was trained using the census.csv database. The target variable was 'salary', which indicates whether an individuals income was less than or equal or greater than 50k. 
The data was split into 80% training and 20% test data using a random seed of 42. The training data was used to fit the encoder, label binarizer, and random forest. 
The categorical features used in preprocessing were "workclass","education," "marital status", occupation", relationship", "race","sex", and "native country".
## Evaluation Data
The evaluation data came from 20% of the test split from the database. The model was evaluated on the full test dataset and on categorical slices of the test dataset. Slice was performace was computed for each unique value in every categorical feature. 
## Metrics
The model was evaluated on using precision,recall, and F1 score. The model achieved the following results:
Precision: .7419
Recall: .6384
F1 Score: .6863

The model was also evaluated on categorical slices. These are included in the slice_output.txt file. 
## Ethical Considerations
This dataset contains demographic and socioeconomic features. The model's slice performance varies across races, sex, and native country. Additional bias analyis should be considered before adopting this model for real-world decision making. 
## Caveats and Recommendations
This model was developed for educational purposes only. 