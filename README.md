# Marcha y enfermedad neurodegenerativa

Clasificación diferencial de enfermedades neurodegenerativas (Parkinson, Huntington, ELA) a partir de la dinámica temporal de la marcha.

**Proyecto Final — Machine Learning · UTEC 2026-1**

---

## Objetivo

El Parkinson, la enfermedad de Huntington y la esclerosis lateral amiotrófica alteran el control motor por mecanismos distintos y producen patrones de marcha distintos. Este proyecto evalúa si descriptores derivados de cinco minutos de marcha instrumentada permiten discriminar **entre** estas patologías, y no solo separar enfermos de sanos.

La tarea es de **clasificación supervisada multiclase** con el sujeto como unidad de análisis (4 clases, 64 sujetos).

### Preguntas de investigación

1. ¿La dinámica de la marcha discrimina entre enfermedades neurodegenerativas, o solo separa enfermos de sanos?
2. ¿Qué magnitud de sesgo optimista introduce evaluar a nivel de zancada en lugar de a nivel de sujeto?
3. ¿La señal discriminante es atribuible a la patología, o la explican confusores demográficos (edad, sexo, velocidad de marcha)?

## Integrantes

- Luis Renato Alvarez Ccopa
- Andersson Chiroque Silva
- David Teofilo Orihuela Serafin
- Jeraldine Dione Rojas Vargas

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
├── src/gait/              Paquete con la lógica reutilizable
│   ├── config.py          Rutas, especificación de datasets, constantes
│   ├── download.py        Descarga y extracción desde PhysioNet
│   ├── gaitndd.py         Carga de la cohorte principal
│   ├── gaitpdb.py         Carga de la cohorte de validación externa
│   ├── quality.py         Reglas de validez fisiológica y reporte de calidad
│   ├── features.py        Descriptores por sujeto y por ventana (CV, DFA, asimetría)
│   └── plots.py           Estilo y componentes de figuras
├── notebooks/
│   ├── 01_eda_gaitndd.ipynb      Análisis exploratorio de la cohorte principal
│   └── 02_cohorte_gaitpdb.ipynb  Caracterización de la cohorte de validación
├── tests/                 Verificación de los descriptores y las reglas de calidad
├── paper/
│   ├── main.tex           Artículo en formato IEEE
│   ├── presentacion.tex   Presentación (Beamer)
│   └── refs.bib
├── reports/figures/       Figuras generadas por los notebooks
├── data/                  Datos crudos (no versionados)
└── requirements.txt
```

Los descriptores no triviales están verificados contra señales de exponente conocido: el estimador de DFA recupera `alpha ≈ 0.5` sobre ruido blanco y `alpha ≈ 1.5` sobre un camino aleatorio.

```bash
pytest tests/
```

## Reproducir el análisis

### Google Colab

Abrir los notebooks de `notebooks/` en Colab y ejecutar todas las celdas. La primera celda clona el repositorio, instala las dependencias y descarga los datos desde PhysioNet.

La primera celda apunta a este repositorio, de modo que funciona sin cambios.

### Entorno local

```bash
git clone https://github.com/jeraldinedione/proyecto-machine-learning.git
cd proyecto-machine-learning

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -c "from sys import path; path.insert(0, 'src'); import gait; gait.download.fetch_all()"
jupyter notebook notebooks/01_eda_gaitndd.ipynb
```

La descarga son 18 MB para `gaitndd` y 302 MB para `gaitpdb`.

## Hallazgos de la entrega P1

**La metadata clínica requiere un parser propio.** El archivo de descripción de sujetos tiene un salto de línea que parte un registro en dos, codifica los faltantes como la cadena literal `MISSING`, etiqueta mal a los 13 sujetos con ELA (`GROUP = "subjects"`) y mezcla tres escalas clínicas incomparables en una sola columna. Un `read_csv` directo devuelve 65 filas en lugar de 64.

**El filtrado de artefactos determina las conclusiones, no solo los números.** Las series se distribuyen sin filtrar y contienen zancadas de hasta 55 segundos. Sin filtrar, el grupo ELA aparenta la marcha más inestable (CV = 0.43, diez veces el control); tras aplicar reglas de validez fisiológica cae a 0.063, y el grupo más inestable pasa a ser Huntington (0.105), que es lo consistente con la fisiopatología coreica.

**El sujeto `hunt20` tiene el canal del pie derecho inutilizable** (zancada mediana de 42.9 s contra 0.99 s del izquierdo, apoyo doble entre −70% y +4317% del ciclo) y se excluye del análisis.

**Particionar por zancada en lugar de por sujeto infla la balanced accuracy en 19 puntos porcentuales** (0.81 contra 0.62, mismos datos y mismo modelo). La estimación agrupada coincide con la evaluación directa a nivel de sujeto (0.64), lo que confirma cuál de las dos es la correcta. Toda la validación de P2 agrupa por sujeto.

**El grupo control es 27 años más joven** que el de Parkinson y mayoritariamente femenino frente a cohortes patológicas mayoritariamente masculinas. Este sesgo no es corregible dentro de `gaitndd` y acota la interpretación de cualquier resultado.

**La cohorte de validación resuelve ese confusor.** En `gaitpdb` la diferencia de edad entre grupos es de 2.6 años y no alcanza significancia (p = 0.063), frente a 27.5 años y p = 3.2 × 10⁻⁵ en la cohorte principal. Eso la convierte en el instrumento para aislar el efecto demográfico, no solo en una prueba de generalización. Trae sus propios defectos, documentados en el notebook 02: tabuladores de relleno variables que rompen `read_csv`, talla en centímetros en uno de los tres sub-estudios, un sujeto huérfano con el identificador mal escrito (`Juc010` por `JuCo10`), y Hoehn & Yahr y UPDRS como fuga perfecta de la etiqueta. Sus 306 grabaciones provienen de 165 sujetos, con hasta siete caminatas por paciente, de modo que exige el mismo particionamiento agrupado.

## Entregas

| Etapa | Contenido | Estado |
|---|---|---|
| P1 | Problema, dataset y EDA | Completa |
| P2 | Preprocesamiento, baseline, modelado, evaluación y discusión | En curso |

## Licencia

Código bajo licencia MIT (ver [`LICENSE`](LICENSE)). Los datos pertenecen a PhysioNet y se rigen por la Open Data Commons Attribution License v1.0; su uso exige citar a los autores originales (ver `data/README.md`).
