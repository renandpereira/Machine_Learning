import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE

# 1. Carregar os dados
data = pd.read_csv('C:/Users/IMILE-TI/Downloads/projeto_de_Regressão/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Pré-processamento
# Converter TotalCharges para numérico e preencher valores ausentes com 0
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)

# Criar nova feature
data['Charges_per_Month'] = data['MonthlyCharges'] / (data['tenure'] + 1)

# Criar a coluna tenure_category
data['tenure_category'] = pd.cut(data['tenure'], bins=[0, 12, 24, data['tenure'].max()],
                                 labels=['Novo', 'Intermediário', 'Antigo'])

# Preencher valores ausentes em tenure_category
data['tenure_category'] = data['tenure_category'].cat.add_categories('0').fillna('0')

# Transformar a variável alvo (Churn) em valores numéricos
data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})

# Codificação de variáveis categóricas
X = pd.get_dummies(data.drop(columns=['customerID', 'Churn']), drop_first=True)
y = data['Churn']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Balanceamento de classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Normalizar variáveis numéricas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Modelagem preditiva
# Regressão Logística
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

# 4. Avaliação dos modelos
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\nModelo: {model.__class__.__name__}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"AUC-ROC: {auc:.4f}")
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

# Avaliar todos os modelos
evaluate_model(lr, X_test, y_test)
evaluate_model(rf, X_test, y_test)
evaluate_model(xgb, X_test, y_test)

# 5. Importância das variáveis (Random Forest)
importances = rf.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
plt.xlabel('Importância')
plt.title('Importância das Variáveis - Random Forest')
plt.show()

# 6. Validação Cruzada
scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\nAUC-ROC Média (Logistic Regression): {scores.mean():.4f}")



