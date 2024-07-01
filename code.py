# Import necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlpack
from sklearn.metrics import *
from sklearn.model_selection import train_test_split # Import train_test_split for splitting data
%matplotlib inline
sns.set(color_codes=True)
# Load loan default prediction (LoanDefault) dataset.
loanData = pd.read_csv("/content/LoanDefault.csv")
# Examine first 5 samples from the dataset.
loanData.head()
# Examine the shape of the dataframe.
print(f"# of rows: {loanData.shape[0]}")
print(f"# of cols: {loanData.shape[1]}")
# Concise summary if all the features in the dataframe.
loanData.info()
## Check the percentage of missing values.
(loanData.isnull().sum() / len(loanData)) * 100
def Resample(data, replace, n_samples, random_state = 123):
 np.random.seed(random_state)
 indices = data.index
 random_sampled_indices = np.random.choice(indices,
 size=n_samples,
 replace=replace)
 return data.loc[random_sampled_indices]
# Oversample the minority class.
negClass = loanData[loanData["Defaulted?"] == 0]
posClass = loanData[loanData["Defaulted?"] == 1]
posOverSampled = Resample(posClass, replace=True, n_samples=len(negClass))
overSampled = pd.concat([negClass, posOverSampled])
# Visualize the distibution of target classes.
sns.countplot(x="Defaulted?", data=overSampled)
plt.show()
# Plot the correlation matrix as heatmap.
PlotHeatMap(overSampled)

# Define FeatureTargetSplit function if it's not defined elsewhere
def FeatureTargetSplit(data, target_col='Defaulted?'):
    features = data.drop(target_col, axis=1)
    target = data[target_col]
    return features, target

features, target = FeatureTargetSplit(overSampled)

# Use train_test_split from sklearn to split the data
Xtrain, Xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=42) # Adjust test_size and random_state as needed

# Create and train Decision Tree model.
output = mlpack.decision_tree(training=Xtrain, labels=ytrain, print_training_accuracy=True)
rf = output["output_model"]
# Predict the values for test data using previously trained model as input.
predictions = mlpack.decision_tree(input_model=rf, test=Xtest)
yPreds = predictions["predictions"].reshape(-1, 1).squeeze()

# Define the modelEval function here
def modelEval(y_true, y_pred):
    '''
    Evaluates the model and prints performance metrics.
    '''
    print("Accuracy:", accuracy_score(y_true, y_pred))
    # Add other metrics as needed (e.g., precision, recall, F1-score)

# Call the modelEval function
modelEval(ytest, yPreds)
