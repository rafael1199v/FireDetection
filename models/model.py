import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

file_name_rf = 'models/randomForestModel_v3'
file_name_lr = 'models/logisticRegressionModel_v1'
file_name_scaler = 'models/scaler_v1'
file_name_imputer = 'models/imputer_v1'

feature_columns = ['elevacion_m', 'humedad_suelo', 'nbr', 'ndvi', 'precip_mm', 'tempC', 'viento_ms']
target_column = 'es_incendio'

df = pd.read_csv("models/bolivia_fires_2023_10k_balanced.csv")

print("="*80)
print("ANÁLISIS EXPLORATORIO DE DATOS".center(80))
print("="*80)
print(f"Total filas: {len(df)}")
print(f"Balance: {df['es_incendio'].mean():.2%} son incendios")
print(f"Clase 0 (No-Incendio): {(df['es_incendio']==0).sum()}")
print(f"Clase 1 (Incendio): {(df['es_incendio']==1).sum()}\n")

X = df[feature_columns]
y = df[target_column]

print("="*80)
print("MATRIZ DE CORRELACIÓN".center(80))
print("="*80)
plt.figure(figsize=(10, 8))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación entre Variables', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)

print("\n" + "="*80)
print("VIF - FACTOR DE INFLACIÓN DE VARIANZA".center(80))
print("="*80)
vif_data = pd.DataFrame()
vif_data["Variable"] = feature_columns
vif_data["VIF"] = [variance_inflation_factor(X_imputed.values, i) for i in range(len(feature_columns))]
vif_data = vif_data.sort_values('VIF', ascending=False)
print(vif_data.to_string(index=False))
print("\nInterpretación: VIF > 10 indica multicolinealidad alta")

plt.figure(figsize=(10, 6))
colors = ['red' if x > 10 else 'green' if x < 5 else 'orange' for x in vif_data['VIF']]
plt.barh(vif_data['Variable'], vif_data['VIF'], color=colors, edgecolor='black')
plt.axvline(x=10, color='red', linestyle='--', linewidth=2, label='VIF = 10 (Alto)')
plt.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='VIF = 5 (Moderado)')
plt.xlabel('Factor de Inflación de Varianza (VIF)', fontsize=12, fontweight='bold')
plt.title('VIF por Variable', fontsize=16, fontweight='bold', pad=20)
plt.legend()
plt.tight_layout()
plt.savefig('outputs/vif_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDatos divididos: Train={len(X_train)} ({y_train.mean():.1%} incendios) | Test={len(X_test)} ({y_test.mean():.1%} incendios)")

print("\n" + "="*80)
print("RANDOM FOREST - ENTRENAMIENTO".center(80))
print("="*80)

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}

rf_base = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

print("Entrenando Random Forest con Grid Search...")
rf_grid.fit(X_train, y_train)

print(f"\nMejores hiperparámetros: {rf_grid.best_params_}")
print(f"Mejor AUC (CV): {rf_grid.best_score_:.4f}")

rf_model = rf_grid.best_estimator_

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)
rf_test_proba = rf_model.predict_proba(X_test)[:, 1]

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)
rf_auc = roc_auc_score(y_test, rf_test_proba)

print(f"\nAccuracy Train: {rf_train_acc:.4f} | Test: {rf_test_acc:.4f} | Diferencia: {rf_train_acc - rf_test_acc:.4f}")
print(f"AUC-ROC: {rf_auc:.4f}")

print("\n" + "="*80)
print("REGRESIÓN LOGÍSTICA - ENTRENAMIENTO".center(80))
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'max_iter': [1000]
}

lr_base = LogisticRegression(random_state=42, class_weight='balanced')
lr_grid = GridSearchCV(lr_base, lr_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)

print("Entrenando Regresión Logística con Grid Search...")
lr_grid.fit(X_train_scaled, y_train)

print(f"\nMejores hiperparámetros: {lr_grid.best_params_}")
print(f"Mejor AUC (CV): {lr_grid.best_score_:.4f}")

lr_model = lr_grid.best_estimator_

lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)
lr_test_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_test_pred)
lr_auc = roc_auc_score(y_test, lr_test_proba)

print(f"\nAccuracy Train: {lr_train_acc:.4f} | Test: {lr_test_acc:.4f}")
print(f"AUC-ROC: {lr_auc:.4f}")

print("\n" + "="*80)
print("VISUALIZACIONES".center(80))
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

cm_rf = confusion_matrix(y_test, rf_test_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], 
            xticklabels=['No-Incendio', 'Incendio'], yticklabels=['No-Incendio', 'Incendio'])
axes[0, 0].set_title(f'Matriz de Confusión - Random Forest\nAUC: {rf_auc:.4f}', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Real', fontsize=12)
axes[0, 0].set_xlabel('Predicción', fontsize=12)

cm_lr = confusion_matrix(y_test, lr_test_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
            xticklabels=['No-Incendio', 'Incendio'], yticklabels=['No-Incendio', 'Incendio'])
axes[0, 1].set_title(f'Matriz de Confusión - Regresión Logística\nAUC: {lr_auc:.4f}', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Real', fontsize=12)
axes[0, 1].set_xlabel('Predicción', fontsize=12)

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_test_proba)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_test_proba)

axes[1, 0].plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'Random Forest (AUC = {rf_auc:.4f})')
axes[1, 0].plot(fpr_lr, tpr_lr, color='green', lw=2, label=f'Reg. Logística (AUC = {lr_auc:.4f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel('Tasa de Falsos Positivos', fontsize=12)
axes[1, 0].set_ylabel('Tasa de Verdaderos Positivos', fontsize=12)
axes[1, 0].set_title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
axes[1, 0].legend(loc="lower right", fontsize=10)
axes[1, 0].grid(alpha=0.3)

feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

axes[1, 1].barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue', edgecolor='black')
axes[1, 1].set_xlabel('Importancia', fontsize=12)
axes[1, 1].set_title('Importancia de Variables - Random Forest', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/model_evaluation_complete.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("COMPARACIÓN FINAL DE MÉTRICAS".center(80))
print("="*80)

comparison_df = pd.DataFrame({
    'Métrica': ['Accuracy Test', 'AUC-ROC', 'True Positives', 'False Negatives', 'Precision', 'Recall'],
    'Random Forest': [
        f"{rf_test_acc:.4f}",
        f"{rf_auc:.4f}",
        cm_rf[1, 1],
        cm_rf[1, 0],
        f"{cm_rf[1,1]/(cm_rf[1,1]+cm_rf[0,1]):.4f}",
        f"{cm_rf[1,1]/(cm_rf[1,1]+cm_rf[1,0]):.4f}"
    ],
    'Reg. Logística': [
        f"{lr_test_acc:.4f}",
        f"{lr_auc:.4f}",
        cm_lr[1, 1],
        cm_lr[1, 0],
        f"{cm_lr[1,1]/(cm_lr[1,1]+cm_lr[0,1]):.4f}",
        f"{cm_lr[1,1]/(cm_lr[1,1]+cm_lr[1,0]):.4f}"
    ]
})

print(comparison_df.to_string(index=False))

print("\n" + "="*80)
print("GUARDANDO MODELOS Y TRANSFORMADORES".center(80))
print("="*80)

pickle.dump(rf_model, open(file_name_rf, 'wb'))
pickle.dump(lr_model, open(file_name_lr, 'wb'))
pickle.dump(scaler, open(file_name_scaler, 'wb'))
pickle.dump(imputer, open(file_name_imputer, 'wb'))

print(f"Random Forest: {file_name_rf}")
print(f"Regresión Logística: {file_name_lr}")
print(f"Scaler: {file_name_scaler}")
print(f"Imputer: {file_name_imputer}")

print("\n" + "="*80)
print("EJEMPLO DE USO PARA PREDICCIONES FUTURAS".center(80))
print("="*80)