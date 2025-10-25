import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def training_model(saved_data):
    X = saved_data.drop('priority', axis=1)
    y = saved_data['priority']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    with open("../Agents/candidate_model.pkl", "wb") as file:
        pickle.dump(clf, file)
    
    print("✅ RandomForest model saved!")
