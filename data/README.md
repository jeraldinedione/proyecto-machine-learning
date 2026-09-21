# Datos

Los datos crudos **no se versionan** en este repositorio: pesan 320 MB y pertenecen a PhysioNet. Esta carpeta documenta cómo obtenerlos y qué contienen.

## Obtención

```python
import sys; sys.path.insert(0, "src")
from gait import config, download

download.fetch(config.GAITNDD)
download.fetch_all()
```

Los archivos quedan en `data/raw/`, que está en `.gitignore`. La descarga es directa desde PhysioNet y no requiere credenciales.

Alternativa manual:

```bash
cd data/raw
curl -O https://physionet.org/static/published-projects/gaitndd/gait-in-neurodegenerative-disease-database-1.0.0.zip
curl -O https://physionet.org/static/published-projects/gaitpdb/gait-in-parkinsons-disease-1.0.0.zip
unzip '*.zip' && rm *.zip
```

## Licencia y condiciones de uso

Ambos datasets se distribuyen bajo **Open Data Commons Attribution License v1.0** (acceso abierto). La licencia permite copiar, distribuir y adaptar los datos con la condición de **atribuir a los autores originales**. No hay restricción de uso académico ni comercial, y no se requiere registro.

Los registros están anonimizados: identifican a cada participante únicamente por un código de grupo y número.

## `gaitndd` — Gait in Neurodegenerative Disease Database

64 sujetos, 17.9 MB. Cohorte principal del proyecto.

| Grupo | Prefijo | n |
|---|---|---|
| Control sano | `control` | 16 |
| Enfermedad de Huntington | `hunt` | 20 |
| Enfermedad de Parkinson | `park` | 15 |
| Esclerosis lateral amiotrófica | `als` | 13 |

**Protocolo.** Cinco minutos de marcha continua a paso cómodo por un pasillo de 77 m, con resistores sensibles a la fuerza bajo cada pie. Señal digitalizada a 300 Hz (90 000 muestras por canal).

**Archivos por sujeto.**

| Archivo | Contenido |
|---|---|
| `<registro>.hea` | Cabecera WFDB: 2 canales, 300 Hz, 90 000 muestras |
| `<registro>.let` | Fuerza bajo el pie izquierdo, formato binario 212 |
| `<registro>.rit` | Fuerza bajo el pie derecho |
| `<registro>.ts` | Serie derivada, 13 columnas por zancada, texto |

**Columnas de los archivos `.ts`**, en orden: tiempo transcurrido (s); intervalo de zancada izquierdo y derecho (s); intervalo de balanceo izquierdo y derecho (s); balanceo izquierdo y derecho (% de la zancada); intervalo de apoyo izquierdo y derecho (s); apoyo izquierdo y derecho (% de la zancada); intervalo de apoyo doble (s); apoyo doble (% de la zancada).

**`subject-description.txt`** — tabulado: identificador, grupo, edad, talla (m), peso (kg), sexo, velocidad de marcha (m/s) y severidad.

> **Advertencia.** La documentación oficial indica que las series `.ts` **no están filtradas**. Contienen artefactos de detección de pisada, incluidos intervalos de zancada de hasta 55 segundos. El módulo `gait.quality` aplica reglas de validez fisiológica; el notebook de EDA documenta el impacto de esa decisión.

### Defectos conocidos de la metadata

Documentados en detalle en `notebooks/01_eda_gaitndd.ipynb`, sección 5.

1. La fila de `hunt20` está partida por un salto de línea; un `read_csv` directo devuelve 65 filas.
2. Los faltantes se codifican como la cadena literal `MISSING` (velocidad de `hunt20`, `als4`, `als5`; peso de `als13`).
3. La columna `GROUP` etiqueta a los 13 sujetos con ELA como `subjects`. La etiqueta confiable es el prefijo del identificador del registro.
4. `Duration/Severity` mezcla cuatro escalas incomparables: Hoehn & Yahr en Parkinson, capacidad funcional total en Huntington, meses desde el diagnóstico en ELA y cero de relleno en controles.
5. Las tallas declaradas son implausibles en los grupos control, Huntington y Parkinson (media de 1.82 m entre mujeres). El grupo ELA usa una convención de unidades distinta, lo que indica que las cohortes se recolectaron por separado.
6. `hunt20` tiene además el canal del pie derecho inutilizable y su registro empieza 42 segundos tarde. Se excluye del análisis.

## `gaitpdb` — Gait in Parkinson's Disease Database

166 sujetos (93 Parkinson, 73 control), 288 MB. Cohorte de validación externa.

**Protocolo.** Aproximadamente dos minutos de marcha a paso cómodo, con 8 sensores de fuerza bajo cada pie (16 canales más 2 señales sumadas), digitalizados a 100 Hz. Reúne tres sub-estudios con protocolos distintos: `Ga` (doble tarea), `Ju` (estimulación auditiva rítmica) y `Si` (caminadora). El sub-estudio es un confusor de protocolo que debe controlarse.

Incluye puntajes **UPDRS** y estadio **Hoehn & Yahr**, lo que habilita una tarea secundaria de regresión de severidad.

## Citas requeridas

Hausdorff, J. M., Lertratanakul, A., Cudkowicz, M. E., Peterson, A. L., Kaliton, D., & Goldberger, A. L. (2000). Dynamic markers of altered gait rhythm in amyotrophic lateral sclerosis. *Journal of Applied Physiology*, 88(6), 2045–2053.

Frenkel-Toledo, S., Giladi, N., Peretz, C., Herman, T., Gruendlinger, L., & Hausdorff, J. M. (2005). Treadmill walking as an external pacemaker to improve gait rhythm and stability in Parkinson's disease. *Movement Disorders*, 20(9), 1109–1114.

Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220.
