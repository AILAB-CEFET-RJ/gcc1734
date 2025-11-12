import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. Carregar e preparar dados ---
X, y = load_wine(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# --- 2. Configurar o modelo ---
clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True, random_state=42)

n_epochs = 50
n_classes = np.unique(y).size

# matriz para armazenar probabilidade da classe correta por época
probs_history = np.zeros((X_train.shape[0], n_epochs))
correct_history = np.zeros((X_train.shape[0], n_epochs))

# --- 3. Treinamento manual por época ---
for epoch in range(n_epochs):
    clf.fit(X_train, y_train)  # warm_start=True mantém pesos anteriores
    probs = clf.predict_proba(X_train)
    
    # probabilidade atribuída à classe correta
    correct_class_probs = probs[np.arange(len(y_train)), y_train]
    probs_history[:, epoch] = correct_class_probs

    # registrar corretude (1 se acertou, 0 caso contrário)
    y_pred = clf.predict(X_train)
    correct_history[:, epoch] = (y_pred == y_train)

    print(f"Época {epoch+1:02d}: acc = {accuracy_score(y_train, y_pred):.3f}")

# --- 4. Calcular métricas do data map ---
confidence = probs_history.mean(axis=1)
variability = probs_history.std(axis=1)
correctness = correct_history.mean(axis=1)

print("Exemplo:", confidence[:5], variability[:5], correctness[:5])
