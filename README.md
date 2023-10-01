# EthicalAIEnforcer
Ensuring responsible AI development and usage in the high-tech world.

# App Code 

To perform sentiment analysis on a given text using a pre-trained AI model, you can follow the code snippet [here](app.py) 

Make sure to replace 'path_to_model.h5' with the actual path to your pre-trained model. This code snippet preprocesses the input text by removing stopwords and special characters, then uses the pre-trained model to classify the sentiment as positive, negative, or neutral. The predicted sentiment is printed as the output.

# Calculate Fairness Metrics

[This code snippet](calculate_fairness_metrics.py) demonstrates the usage of the fairness evaluation metric. It defines a function calculate_fairness_metrics that takes a dataset and the predictions made by an AI model as inputs. The dataset should contain the true labels and the indices of the protected group. The function calculates and returns three fairness metrics: disparate impact, equal opportunity difference, and statistical parity difference.

To use the code, you need to provide the dataset and predictions as input arguments to the calculate_fairness_metrics function. In the usage example, a sample dataset and predictions are provided. The function then calculates the fairness metrics and prints them as output.
