# Marcha y enfermedad neurodegenerativa

Clasificación de enfermedades neurodegenerativas (Parkinson, Huntington, ELA) a partir de la dinámica temporal de la marcha.

**Proyecto Final — Machine Learning · UTEC 2026-1**

---

## Objetivo

El Parkinson, la enfermedad de Huntington y la esclerosis lateral amiotrófica alteran el control motor por mecanismos distintos y producen patrones de marcha distintos. Este proyecto evalúa si descriptores derivados de cinco minutos de marcha instrumentada permiten discriminar **entre** estas patologías, y no solo separar enfermos de sanos.

La tarea es de **clasificación supervisada multiclase** con el sujeto como unidad de análisis (4 clases, 63 sujetos tras el control de calidad).

### Preguntas de investigación

1. ¿La dinámica de la marcha discrimina entre enfermedades neurodegenerativas, o solo separa enfermos de sanos?
2. ¿Qué magnitud de sesgo optimista introduce evaluar a nivel de zancada en lugar de a nivel de sujeto?
3. ¿La señal discriminante es atribuible a la patología, o la explican confusores demográficos (edad, sexo, velocidad de marcha)?

## Integrantes

- Luis Renato Alvarez Ccopa
- Andersson Chiroque Silva
- David Teofilo Orihuela Serafin
- Jeraldine Dione Rojas Vargas
- Dayanna Gisell Quispe Gallardo

Universidad de Ingeniería y Tecnología — Lima, Perú

## Datos

| Dataset | Rol | Sujetos | Fuente |
|---|---|---|---|
| Gait in Neurodegenerative Disease Database | Cohorte principal | 64 (15 Parkinson, 20 Huntington, 13 ELA, 16 control) | [PhysioNet `gaitndd`](https://physionet.org/content/gaitndd/1.0.0/) |
| Gait in Parkinson's Disease Database | Validación externa | 166 (93 Parkinson, 73 control) | [PhysioNet `gaitpdb`](https://physionet.org/content/gaitpdb/1.0.0/) |

Ambos se distribuyen bajo licencia **Open Data Commons Attribution v1.0** y son de acceso abierto. Los datos **no se versionan en este repositorio**; se descargan con el código incluido. Ver [`data/README.md`](data/README.md).

## Estructura del repositorio

```
.
├── src/gait/                    Paquete con la lógica reutilizable
│   ├── config.py                Rutas, datasets, listas de descriptores y constantes
│   ├── download.py              Descarga y extracción desde PhysioNet
│   ├── gaitndd.py               Carga de la cohorte principal
│   ├── gaitpdb.py               Carga de la cohorte externa y extracción de zancadas desde la fuerza plantar
│   ├── quality.py               Reglas de validez fisiológica y reporte de calidad
│   ├── features.py              Descriptores por sujeto y por ventana (CV, DFA, entropía muestral, ...)
│   ├── cohort.py                Filtrado y construcción de las tablas de modelado
│   ├── models.py                Referencias, modelos y espacios de hiperparámetros
│   ├── evaluation.py            Validación cruzada anidada, métricas y pruebas estadísticas
│   └── plots.py                 Estilo y componentes de figuras
├── notebooks/
│   ├── 01_eda_gaitndd.ipynb           P1 · Análisis exploratorio de la cohorte principal
│   ├── 02_cohorte_gaitpdb.ipynb       P1 · Caracterización de la cohorte externa
│   ├── 03_preprocesamiento.ipynb      P2 · Filtrado, descriptores y variables excluidas
│   ├── 04_modelado.ipynb              P2 · Referencias, modelos, validación anidada y comparación
│   ├── 05_interpretacion.ipynb        P2 · Errores, importancia, confusores y sensibilidad
│   └── 06_validacion_externa.ipynb    P2 · Transporte del modelo a gaitpdb
├── scripts/
│   ├── descargar_datos.py       Descarga ambas cohortes
│   └── ejecutar_notebooks.py    Ejecuta los notebooks sin interfaz y guarda sus salidas
├── tests/                       Pruebas de descriptores, calidad, extracción de zancadas y evaluación
├── paper/
│   ├── p1/                      Artículo y presentación de la entrega 1
│   └── p2/                      Artículo y presentación de la entrega 2
├── reports/
│   ├── figures/                 Figuras generadas por los notebooks
│   └── tables/                  Resultados en CSV generados por los notebooks
├── data/                        Datos crudos y procesados (no versionados)
└── requirements.txt
```

## Reproducir el análisis

### Entorno local

```bash
git clone https://github.com/jeraldinedione/proyecto-machine-learning.git
cd proyecto-machine-learning

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/descargar_datos.py
pytest tests/
python scripts/ejecutar_notebooks.py
```

La descarga son 18 MB para `gaitndd` y 302 MB para `gaitpdb`. Si en macOS la descarga falla con `CERTIFICATE_VERIFY_FAILED`, falta instalar los certificados del Python de python.org (`Install Certificates.command`) o se puede descargar a mano con `curl` (ver `data/README.md`).

`ejecutar_notebooks.py` corre los seis notebooks en orden y sobrescribe sus salidas, las figuras de `reports/figures/` y las tablas de `reports/tables/`. También acepta nombres concretos:

```bash
python scripts/ejecutar_notebooks.py 04_modelado.ipynb
```

Todas las particiones usan la semilla 42 (`config.RANDOM_STATE`), así que los resultados se reproducen exactamente con las mismas versiones de las librerías.

### Google Colab

Abrir cualquier notebook de `notebooks/` en Colab y ejecutar todas las celdas. La primera celda clona el repositorio, instala las dependencias y la segunda descarga los datos desde PhysioNet.

### Tiempo de ejecución

En un portátil con 10 núcleos: notebook 03, menos de un minuto; 04, unos 12 minutos; 05, unos 16; 06, unos 7. La validación cruzada anidada usa todos los núcleos disponibles.

## Hallazgos de la entrega P1

**La metadata clínica requiere un parser propio.** El archivo de descripción de sujetos tiene un salto de línea que parte un registro en dos, codifica los faltantes como la cadena literal `MISSING`, etiqueta mal a los 13 sujetos con ELA (`GROUP = "subjects"`) y mezcla tres escalas clínicas incomparables en una sola columna. Un `read_csv` directo devuelve 65 filas en lugar de 64.

**El filtrado de artefactos determina las conclusiones, no solo los números.** Las series se distribuyen sin filtrar y contienen zancadas de hasta 55 segundos. Sin filtrar, el grupo ELA aparenta la marcha más inestable (CV = 0.43, diez veces el control); tras aplicar reglas de validez fisiológica cae a 0.063, y el grupo más inestable pasa a ser Huntington (0.105), que es lo consistente con la fisiopatología coreica.

**El sujeto `hunt20` tiene el canal del pie derecho inutilizable** (zancada mediana de 42.9 s contra 0.99 s del izquierdo, apoyo doble entre −70% y +4317% del ciclo) y se excluye del análisis.

**Particionar por zancada en lugar de por sujeto infla la balanced accuracy en 19 puntos porcentuales** (0.81 contra 0.62, mismos datos y mismo modelo). La estimación agrupada coincide con la evaluación directa a nivel de sujeto (0.64), lo que confirma cuál de las dos es la correcta. Toda la validación de P2 agrupa por sujeto.

**El grupo control es 27 años más joven** que el de Parkinson y mayoritariamente femenino frente a cohortes patológicas mayoritariamente masculinas. Este sesgo no es corregible dentro de `gaitndd` y acota la interpretación de cualquier resultado.

**La cohorte de validación resuelve ese confusor.** En `gaitpdb` la diferencia de edad entre grupos es de 2.6 años y no alcanza significancia (p = 0.063), frente a 27.5 años y p = 3.2 × 10⁻⁵ en la cohorte principal. Eso la convierte en el instrumento para aislar el efecto demográfico, no solo en una prueba de generalización. Trae sus propios defectos, documentados en el notebook 02: tabuladores de relleno variables que rompen `read_csv`, talla en centímetros en uno de los tres sub-estudios, un sujeto huérfano con el identificador mal escrito (`Juc010` por `JuCo10`), y Hoehn & Yahr y UPDRS como fuga perfecta de la etiqueta. Sus 306 grabaciones provienen de 165 sujetos, con hasta siete caminatas por paciente, de modo que exige el mismo particionamiento agrupado.

## Resultados de la entrega P2

Validación cruzada anidada y agrupada por sujeto (5 particiones × 10 repeticiones, 63 sujetos, 12 descriptores de marcha). Media ± desviación estándar entre repeticiones.

| Modelo | Exactitud balanceada | F1 macro | AUC |
|---|---|---|---|
| Clase mayoritaria | 0.250 ± 0.000 | 0.116 ± 0.000 | 0.500 |
| Logística con CV y DFA | 0.538 ± 0.034 | 0.519 ± 0.036 | 0.735 |
| Regresión logística | 0.591 ± 0.041 | 0.583 ± 0.045 | 0.812 |
| SVM (RBF) | 0.537 ± 0.026 | 0.508 ± 0.025 | 0.783 |
| Bosque aleatorio | 0.605 ± 0.016 | 0.595 ± 0.018 | 0.829 |
| Gradient boosting | 0.566 ± 0.049 | 0.560 ± 0.053 | 0.763 |
| Bosque entrenado con ventanas | 0.647 ± 0.028 | 0.631 ± 0.029 | 0.862 |

- **PI1.** Separar control de paciente alcanza 0.86–0.87 de exactitud balanceada; distinguir entre Parkinson, Huntington y ELA, 0.56–0.62 (azar 0.33). El par más difícil es Parkinson–Huntington (AUC 0.70), y los errores en Huntington se concentran en los casos leves.
- **PI2.** Evaluar ventanas como independientes infla la exactitud en 19 puntos (P1). Entrenar con ventanas y evaluar por sujeto es válido y dio el mejor resultado, aunque la mejora no es significativa.
- **PI3.** Edad y sexo solos clasifican con 0.44–0.46 dentro de `gaitndd`. Transportado sin reentrenar a `gaitpdb`, donde la edad casi no difiere entre grupos, el modelo de tiempos de zancada conserva un AUC de 0.71 [0.63, 0.79] y el de edad cae a 0.52.
- Ninguna diferencia entre modelos es significativa (prueba t corregida de Nadeau y Bengio); todos superan el azar (permutación, p = 0.001).
- Las variables que dependen de la duración del registro (número de zancadas, duración observada, cadencia) mejoran la validación interna pero hacen que la logística clasifique al 98 % de `gaitpdb` como control. Por eso quedan fuera del modelo.

Detalle completo en `paper/p2/main.pdf` y en los notebooks 03 a 06; todas las tablas están en `reports/tables/`.

## Entregas

| Etapa | Contenido | Documentos |
|---|---|---|
| P1 | Problema, dataset y EDA | `paper/p1/` · notebooks 01 y 02 |
| P2 | Preprocesamiento, baseline, modelado, evaluación y discusión | `paper/p2/` · notebooks 03 a 06 |

## Licencia

Código bajo licencia MIT (ver [`LICENSE`](LICENSE)). Los datos pertenecen a PhysioNet y se rigen por la Open Data Commons Attribution License v1.0; su uso exige citar a los autores originales (ver `data/README.md`).
