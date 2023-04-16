import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 导入决策树算法
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#标准化
from sklearn.preprocessing import StandardScaler
#导入pca
from sklearn.decomposition import PCA as sklearnPCA

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None

    def fit(self, X):
        # 计算协方差矩阵
        covariance_matrix = np.cov(X.T)
        # 计算协方差矩阵的特征值和特征向量
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        # 倒序排列特征值、特征向量
        eigen_values_sorted = np.argsort(eigen_values)[::-1]
        eigen_vectors_sorted = eigen_vectors[:, eigen_values_sorted]
        # 选择特征值大小前n的特征向量
        self.components = eigen_vectors_sorted[:, :self.n_components]

    def transform(self, X):
        return np.dot(X, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


dataset = pd.read_csv('Data_Cortex_Nuclear.csv')
X = dataset.iloc[:, 1:78].values
Y = dataset.iloc[:, -1].values
# 将缺失值替换为0
X = np.nan_to_num(X)
X_sklearn = X
# 对x进行pca降维
pca = PCA(n_components=5)
X = pca.fit_transform(X)
sklearn_pca = sklearnPCA(n_components=5)
X_sklearn = sklearn_pca.fit_transform(X_sklearn)
#实现的pca
# 取出前90%的数据作为训练集
X_train = X[:int(len(X) * 0.9)] 
Y_train = Y[:int(len(Y) * 0.9)]
# 取出后10%的数据作为测试集
X_test = X[int(len(X) * 0.9):]
Y_test = Y[int(len(Y) * 0.9):]
# 训练决策树模型
classifier = DecisionTreeClassifier(random_state=234)
classifier.fit(X_train, Y_train)
# 预测结果
Y_pred = classifier.predict(X_test)
# 计算准确率
print(accuracy_score(Y_test, Y_pred))


#sklearn的pca
# 取出前60%的数据作为训练集
X_train = X_sklearn[:int(len(X_sklearn) * 0.9)]
Y_train = Y[:int(len(Y) * 0.9)]
# 取出后40%的数据作为测试集
X_test = X_sklearn[int(len(X_sklearn) * 0.9):]
Y_test = Y[int(len(Y) * 0.9):]
# 训练决策树模型
classifier = DecisionTreeClassifier(random_state=234)
classifier.fit(X_train, Y_train)
# 预测结果
Y_pred = classifier.predict(X_test)
# 计算准确率
print(accuracy_score(Y_test, Y_pred))

