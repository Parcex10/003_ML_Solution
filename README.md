# 003_ML_Solution
#### Autores: Adrián Herrera, Patrick F. Barcena y Carlos Moreno

# 📈 ML Pipeline para Predicción de Señales de Trading (SPY ETF)

Este proyecto implementa un pipeline de aprendizaje automático para predecir señales de trading en el ETF SPY utilizando regresión logística. La solución integra indicadores técnicos, variables categóricas relacionadas con el mercado, técnicas de balanceo de clases y optimización mediante validación cruzada.

---

## 🎯 Objetivo

Construir un modelo capaz de predecir si el retorno del activo será positivo (`Buy`) o negativo (`No Buy`), basado en información técnica y contextual.

---

## 🧪 Resumen del Experimento

Se entrenaron dos versiones del modelo:

- **Modelo Base:** Incluye preprocesamiento, balanceo con SMOTE y LogisticRegression.
- **Modelo Avanzado:** Integra PCA para reducción de dimensionalidad y GridSearchCV para ajuste de hiperparámetros.

Ambos modelos fueron evaluados con métricas de clasificación como accuracy, precision, recall, F1-score y matriz de confusión.

---

## 🧠 Indicadores Técnicos Usados

- Media Móvil Simple (SMA)
- Media Móvil Exponencial (EWMA)
- Volatilidad Móvil
- RSI (Relative Strength Index)
- Cambio de Volumen
- Bandas de Bollinger

---

## ⚙️ Tecnologías Usadas

- Python 3.x
- Pandas / NumPy / Matplotlib / Seaborn
- scikit-learn
- imbalanced-learn
- Jupyter Notebooks

---

## 🗂️ Estructura del Proyecto

003_ML_Solution/
├── data/ # `SPY_dataset_project.csv`


├── notebooks/ # Notebooks base + reporte final llamado `Report_and_Solution.ipynb`


├── README.md # Introducción y contexto


├── requirements.txt # Dependencias


└── utils.py # (opcional) Funciones personalizadas


---

## 📊 Resultados

| Métrica            | Modelo Base | PCA + Optimización |
|--------------------|-------------|--------------------|
| Accuracy           | 0.96        | 0.95               |
| F1-score           | 0.96        | 0.96               |
| Falsos Positivos   | 5           | 4                  |
| Falsos Negativos   | 1           | 3                  |

> Ambos modelos generalizan bien. El uso de PCA permitió mantener el rendimiento con menor cantidad de componentes.

---

## 🤔 Reflexión Final

Durante el desarrollo de este proyecto, como equipo tuvimos la oportunidad de construir paso a paso un pipeline completo de machine learning para predecir señales de trading en el ETF SPY. La experiencia nos permitió reforzar no solo conceptos técnicos, sino también entender mejor cómo se comportan los modelos frente a datos financieros reales.

Uno de los mayores retos fue encontrar el equilibrio entre un modelo que sea preciso, pero que al mismo tiempo no sobreajuste ni dependa de demasiadas variables. Fue ahí donde aplicar técnicas como SMOTE para el balanceo de clases, o PCA para reducir dimensionalidad, resultaron fundamentales. La exploración de los indicadores técnicos también fue clave para enriquecer nuestras features desde el inicio.

Lo que más resaltó fue ver cómo incluso con un modelo sencillo como la regresión logística, se puede lograr un desempeño robusto si se estructura correctamente el pipeline. Los resultados hablaron por sí mismos: altos valores de F1-score, baja cantidad de errores, y un modelo que generaliza bien. Nos llevamos no solo un aprendizaje técnico, sino también una base sólida para futuros proyectos de predicción financiera.

---

## 🚀 Cómo ejecutar

1. Clona el repositorio:

   git clone https://github.com/Parcexx10/003_ML_Solution.git
   cd 003_ML_Solution

2. Instala dependencias:

pip install -r requirements.txt

3. Abre el notebook principal

jupyter notebook Report_and_Solution.ipynb





