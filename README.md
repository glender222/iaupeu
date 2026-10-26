# Entorno de Trabajo de Data Science e Inteligencia Artificial (IA)

Bienvenido al entorno de desarrollo para el proyecto de análisis de fraude financiero y modelado de Machine Learning (`bank_fraud.csv`).

---

## 🛠️ 1. Arquitectura del Entorno

- **Versión de Python:** Python `3.12` gestionado con `uv` (alto rendimiento y reproducibilidad).
- **Aceleración por Hardware:** **PyTorch 2.6+cu124** con soporte nativo para **CUDA 12.4** (NVIDIA GeForce RTX 3050 Laptop GPU).
- **Kernel de Jupyter:** Registrado como `Python 3.12 (IA & Data - iaupeu)`.

---

## 📦 2. Librerías Instaladas

| Categoría | Librerías Principales | Propósito |
|---|---|---|
| **Manipulación & Alto Rendimiento** | `polars`, `pandas`, `pyarrow`, `numpy`, `scipy` | Procesamiento ultrarrápido de 1M+ filas en `bank_fraud.csv`. |
| **Visualización** | `matplotlib`, `seaborn`, `plotly` | Gráficos estáticos e interactivos para análisis exploratorio. |
| **Machine Learning & Modelado** | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` | Algoritmos de clasificación supervisada de estado del arte. |
| **Datos Desbalanceados** | `imbalanced-learn` (`imblearn`) | Técnicas SMOTE, ADASYN, submuestreo para fraude minoritario. |
| **Explicabilidad & Métricas** | `shap` | Valores SHAP para interpretabilidad del modelo. |
| **Deep Learning** | `torch`, `torchvision`, `torchaudio` | Redes neuronales aceleradas por GPU. |
| **Entorno Interactivo** | `ipykernel`, `jupyterlab`, `tqdm`, `rich` | Notebooks, barras de progreso y consola enriquecida. |
| **Calidad de Código** | `ruff` | Linter y formateador ultrarrápido integrado en VS Code. |

---

## 🧩 3. Extensiones y Plugins de VS Code

Las siguientes extensiones clave han sido configuradas:

1. **Jupyter** (`ms-toolsai.jupyter`):
   - Ejecución interactiva de celdas `.ipynb`.
   - Soporte para variables interactivas y depuración de celdas.

2. **Microsoft Data Wrangler** (`ms-toolsai.datawrangler`):
   - **Esencial para Data Science en VS Code:** Abre dataframes de Pandas/Polars en una interfaz visual interactiva con estadísticas automáticas, detección de outliers, correlaciones y generación de código Python en 1 clic.

3. **Rainbow CSV** (`mechatroner.rainbow-csv`):
   - Colorea cada columna de `bank_fraud.csv` con colores diferentes para inspeccionar el dataset al instante. Permite consultas SQL sobre archivos CSV con RBQL.

4. **Ruff** (`charliermarsh.ruff`):
   - Formateador automático al guardar archivos `.py`, ordenación de imports y detección de errores de sintaxis en milisegundos.

5. **Python & Pylance** (`ms-python.python`, `ms-python.vscode-pylance`):
   - Autocompletado inteligente, tipado estático y navegación de código.

6. **Extensiones de IA recomendadas:**
   - **Claude Code** (`anthropic.claude-code`) o **GitHub Copilot** (`github.copilot` / `github.copilot-chat`): Asistente de autocompletado y chat en línea.
   - **Continue** (`Continue.continue`): Si deseas conectar modelos locales (Ollama/LM Studio) o APIs abiertas.

---

## 🚀 4. Guía Rápida de Uso

### Activar el Entorno en la Terminal
```bash
source .venv/bin/activate
```

### Probar y Verificar el Entorno
Ejecuta el script de diagnóstico que valida librerías, GPU CUDA y el dataset `bank_fraud.csv`:
```bash
python verify_env.py
```

### Trabajar con Notebooks en VS Code
1. Abre o crea cualquier archivo `.ipynb` (ej. `01_exploracion_fraude.ipynb`).
2. En la esquina superior derecha del notebook, haz clic en **Select Kernel** -> **Python Environments...**
3. Elige: **`Python 3.12 (IA & Data - iaupeu)`** (o `.venv/bin/python`).
4. ¡Listo! Todas las celdas se ejecutarán utilizando la GPU y las librerías instaladas.

### Uso de Microsoft Data Wrangler
- En cualquier script o celda de Jupyter donde tengas un DataFrame (`df`):
  En la pestaña **Jupyter: Variables**, haz clic en el icono de Data Wrangler junto al DataFrame para abrir el visor interactivo de datos.
