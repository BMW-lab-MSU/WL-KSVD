## Example script for basic useage of WL+KSVD method

# Load data
from karateclub.dataset import GraphSetReader

reader = GraphSetReader("reddit10k")

graphs = reader.get_graphs()
y = reader.get_target()

# Fit the WL+KSVD model
from WL_KSVD import WL_KSVD

model = WL_KSVD()
model.fit(graphs)
X = model.get_embedding()

# Train Test divide
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying ML model
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
y_hat = downstream_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_hat)
print('AUC: {:.4f}'.format(auc))
print('Done')
