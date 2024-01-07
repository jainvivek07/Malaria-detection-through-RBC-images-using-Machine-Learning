from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def LogisticModel(x_train,x_test,y_train,y_test):
    model = LogisticRegression(max_iter = 500)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # score = model.score(x_test, y_test)
    # return score
    return accuracy_score(y_pred, y_test)
    

def SVMModel(x_train,x_test,y_train,y_test):
    clf_svc = SVC().fit(x_train, y_train)
    y_pred = clf_svc.predict(x_test)
    return accuracy_score(y_test, y_pred)

def DTModel(x_train,x_test,y_train,y_test):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(x_train, y_train)
    y_pred = dt_model.predict(x_test)
    return accuracy_score(y_pred, y_test)

def StandardScaling(X,Y):
    standard = StandardScaler()
    standard = standard.fit(X)
    return standard.transform(Y)

def random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

def naive_bayes(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    return accuracy_score(y_test, y_pred)

def ada_boost(X_train, X_test, y_train, y_test):
    ab = AdaBoostClassifier()
    ab.fit(X_train, y_train)
    y_pred = ab.predict(X_test)
    return accuracy_score(y_test, y_pred)
    
def apply_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

def apply_lda(X_train, X_test, y_train, y_test, n_components=None):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda
