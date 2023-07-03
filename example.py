## Example script for basic useage of WL+KSVD method
# Following the example by :

# Load data
from karateclub.dataset import GraphSetReader

print('Loading data')
reader = GraphSetReader("reddit10k")

graphs = reader.get_graphs()
y = reader.get_target()

# Fit the WL+KSVD model
from WL_KSVD import WL_KSVD

print('Fitting the embedding model')
model = WL_KSVD()
model.fit(graphs)
X = model.get_embedding()

# Train Test divide
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying ML model
print('Applying ML model')
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
y_hat = downstream_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_hat)
print('Without overfit protection')
print('AUC: {:.4f}'.format(auc))


print('Overfit protection')
# To avoid overfitting we can fit the WL+KSVD model on the training set and infer the embedding of the test set.
# Fisrt we divide the data into train and test sets.

G_train, G_test, y_train, y_test = train_test_split(graphs, y, test_size=0.2, random_state=42)

# divide the train set further into vocab training and ML training sets

G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(G_train, y_train, test_size=0.75, random_state=42)

# Fit the WL+KSVD model on the vocab training set
print('Fitting the embedding model')
model = WL_KSVD()
model.fit(G_vocab_train)
X_vocab_train = model.get_embedding()

# Infer the embedding of the ML training set
X_ML_train = model.infer(G_ML_train)
X_ML_test = model.infer(G_test)

# Applying ML model
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

print('Applying ML model')
downstream_model = LogisticRegression(random_state=0).fit(X_ML_train, y_ML_train)
y_ML_hat = downstream_model.predict_proba(X_ML_test)[:, 1]
auc = roc_auc_score(y_test, y_ML_hat)
print('With overfit protection')
print('AUC: {:.4f}'.format(auc))
print('Done')


