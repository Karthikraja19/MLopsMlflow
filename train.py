
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score,precision_score,recall_score,f1_score
import mlflow
import mlflow.sklearn
import joblib
from sklearn.svm import SVC

data = pd.read_csv('data\Housing.csv')
data.replace({'yes': 1, 'no': 0}, inplace=True)
data.replace({'furnished': 1, 'semi-furnished': 0, 'unfurnished': 2}, inplace=True)

data.head()

x=data.drop(['price'],axis=1)
y=data.price.copy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

mlflow.set_experiment("housing-experiment-Random forest")

with mlflow.start_run():

    n_estimators = 20
    random_state = 42

    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('random_state', random_state)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse_rf = mean_squared_error(y_test, y_pred)
    r2_rf = r2_score(y_test, y_pred)
    accuracy_rf = accuracy_score(y_test, y_pred)
    precision_rf = precision_score(y_test, y_pred, average='macro')
    recall_rf = recall_score(y_test, y_pred, average='macro')
    f1_rf = f1_score(y_test, y_pred, average='macro')

    mlflow.log_metric('mse_rf', mse_rf)
    mlflow.log_metric('r2_rf', r2_rf)
    mlflow.log_metric('accuracy_rf', accuracy_rf)
    mlflow.log_metric('precision_rf', precision_rf)
    mlflow.log_metric('recall_rf',recall_rf)
    mlflow.log_metric('f1_rf',f1_rf)

    print(f"mse_rf: {mse_rf}")
    print(f"r2_rf: {r2_rf}")
    print(f"accuracy_rf: {accuracy_rf}")
    print(f"precision_rf: {precision_rf}")
    print(f"recall_rf: {recall_rf}") 
    print(f"f1_rf: {f1_rf}")  

    mlflow.sklearn.log_model(model, 'random-forest-model')
    joblib.dump(model, 'random-forest-model.pkl')
    mlflow.log_artifact('random-forest-model.pkl')

mlflow.set_experiment("housing-experiment - Decision Tree")

with mlflow.start_run():

    # Initialize and train the model
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    # Make predictions
    y_pred = clf.predict(x_test)

    # Calculate metrics
    mse_dt = mean_squared_error(y_test, y_pred)
    r2_dt = r2_score(y_test, y_pred)
    accuracy_dt = accuracy_score(y_test, y_pred)
    precision_dt = precision_score(y_test, y_pred, average='macro')
    recall_dt = recall_score(y_test, y_pred, average='macro')
    f1_dt = f1_score(y_test, y_pred, average='macro')

    # Log metrics
    mlflow.log_metric('mse_dt', mse_dt)
    mlflow.log_metric('r2_dt', r2_dt)
    mlflow.log_metric('accuracy_dt', accuracy_dt)
    mlflow.log_metric('precision_dt', precision_dt)
    mlflow.log_metric('recall_dt', recall_dt)
    mlflow.log_metric('f1_dt', f1_dt)

    #print metrics
    print(f"mse_dt: {mse_dt}")
    print(f"r2_dt: {r2_dt}")
    print(f"accuracy_dt: {accuracy_dt}")
    print(f"precision_dt: {precision_dt}")
    print(f"recall_dt: {recall_dt}")
    print(f"f1_dt: {f1_dt}") 

    mlflow.sklearn.log_model(clf, 'decision-tree-model')
    joblib.dump(clf, 'decision-tree-model.pkl')
    mlflow.log_artifact('decision-tree-model.pkl')
    

# Set the MLflow experiment
mlflow.set_experiment("housing-experiment - SVM")

with mlflow.start_run():
    # Initialize and train the model
    clf_svm = SVC()
    clf_svm.fit(x_train, y_train)

    # Make predictions
    y_pred_svm = clf_svm.predict(x_test)


    # Calculate metrics
    mse_svm = mean_squared_error(y_test, y_pred)
    r2_svm = r2_score(y_test, y_pred)
    accuracy_svm = accuracy_score(y_test, y_pred)
    precision_svm = precision_score(y_test, y_pred, average='macro')
    recall_svm = recall_score(y_test, y_pred, average='macro')
    f1_svm = f1_score(y_test, y_pred, average='macro')

    # Log metrics
    mlflow.log_metric('mse_svm', mse_svm)
    mlflow.log_metric('r2_svm', r2_svm)
    mlflow.log_metric('accuracy_svm', accuracy_svm)
    mlflow.log_metric('precision_svm', precision_svm)
    mlflow.log_metric('recall_svm', recall_svm)
    mlflow.log_metric('f1_svm', f1_svm)

    #print metrics
    print(f"mse_svm: {mse_svm}")
    print(f"r2_svm: {r2_svm}")
    print(f"accuracy_svm: {accuracy_svm}")
    print(f"precision_svm:{precision_svm}")
    print(f"recall_svm: {recall_svm}")
    print(f"f1_svm: {f1_svm}") 

    # Log the model to MLflow and save it locally
    mlflow.sklearn.log_model(clf_svm, 'svm-model')
    joblib.dump(clf_svm, 'svm-model.pkl')
    mlflow.log_artifact('svm-model.pkl')