import numpy as np

class NaiveBayesDoZero:
    def __init__(self):
        self.prior = {}
        self.condicional = {}
        self.classes = []
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = len(y)
        for c in self.classes:
            X_c = X[y == c]
            self.prior[c] = len(X_c) / n_samples
            self.condicional[c] = {}
            for col in X.columns:
                valores = X[col].unique()
                self.condicional[c][col] = {}
                for v in valores:
                    # Laplace: (contagem + 1) / (total da classe + num de valores possiveis)
                    contagem = (X_c[col] == v).sum()
                    self.condicional[c][col][v] = (contagem + 1) / (len(X_c) + len(valores))
                    
    def predict(self, X):
        preds = []
        for _, row in X.iterrows():
            prob_c = {}
            for c in self.classes:
                prob = np.log(self.prior[c]) # Log evita underflow
                for col in X.columns:
                    v = row[col]
                    if v in self.condicional[c][col]:
                        prob += np.log(self.condicional[c][col][v])
                    else:
                        prob += np.log(1e-6)
                prob_c[c] = prob
            preds.append(max(prob_c, key=prob_c.get))
        return preds

    def predict_proba(self, X):
        scores = []
        for _, row in X.iterrows():
            scores.append([np.log(self.prior[c]) + sum(np.log(self.condicional[c][col].get(row[col], 1e-6)) for col in X.columns) for c in self.classes])
        scores = np.asarray(scores)
        scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return scores / scores.sum(axis=1, keepdims=True)
