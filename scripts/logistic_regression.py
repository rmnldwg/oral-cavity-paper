#logistic regression
from pathlib import Path
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MPLSTYLE = Path("./scripts/.mplstyle")

usz_blue = '#005ea8'
usz_green = '#00afa5'
usz_red = '#ae0060'
usz_orange = '#f17900'
usz_gray = '#c5d5db'

plt.style.use(MPLSTYLE)

reg_data_raw = pd.read_csv("./data/lymph_nodes_invest_OC.csv", sep=";")
reg_data = reg_data_raw.iloc[:,[-6,-2,-1]].dropna()
classification_data = reg_data_raw.iloc[:,-1].dropna()

# Create the table data
data = {
    '': ['Clinically Positive', 'Clinically Negative'],
    'Pathology Positive': [sum(classification_data=="tp"), sum(classification_data=="fn")],
    'Pathology Negative': [sum(classification_data=="fp"), sum(classification_data=="tn")]
}

sns.histplot(x=reg_data.iloc[:,0], hue=reg_data.iloc[:,2], multiple="stack", palette=[usz_orange, usz_red, usz_green], alpha=0.6)



"""# Assuming reg_data is your DataFrame
fig, ax = plt.subplots()

# Select the data for the histogram
x_data = reg_data.iloc[:, 0]
categories = reg_data.iloc[:, 2].unique()

# Increase the bandwidth by adjusting the bins parameter
bins = 10  # Adjust the bins value as needed

# Plot the histogram for each category
for category in categories:
    # Filter data for the current category
    filtered_data = x_data[reg_data.iloc[:, 2] == category]
    
    # Plot histogram for the current category
    ax.hist(filtered_data, bins=bins, label=category, stacked=True, alpha=0.6)

# Set labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Frequency')
ax.set_title('Stacked Histogram')

# Add a legend
ax.legend()
"""
# Display the plot
#plt.show()



# Create the DataFrame
df = pd.DataFrame(data).set_index('')

# Create the table plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center', colWidths=[0.2, 0.2, 0.2])

# Save the table as a PNG
plt.savefig('./figures/table_errors.png')


#logistic regression
X = pd.DataFrame(reg_data.iloc[:,0])
X = X.values.reshape((-1,1))
y = reg_data.iloc[:,1]

# split X and y into training and testing sets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()