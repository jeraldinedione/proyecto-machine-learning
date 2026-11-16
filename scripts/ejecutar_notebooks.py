from __future__ import annotations

import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
CUADERNOS = RAIZ / "notebooks"


def ejecutar(ruta: Path, tiempo_maximo: int) -> None:
    cuaderno = nbformat.read(ruta, as_version=4)
    cliente = NotebookClient(
        cuaderno,
        timeout=tiempo_maximo,
        kernel_name="python3",
        resources={"metadata": {"path": str(CUADERNOS)}},
    )
    cliente.execute()
    nbformat.write(cuaderno, ruta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta los cuadernos del proyecto y guarda sus salidas.")
    parser.add_argument("cuadernos", nargs="*", help="nombres de archivo dentro de notebooks/; por defecto, todos")
    parser.add_argument("--tiempo-maximo", type=int, default=3600, help="segundos por celda")
    argumentos = parser.parse_args()

    rutas = [CUADERNOS / nombre for nombre in argumentos.cuadernos] or sorted(CUADERNOS.glob("*.ipynb"))
    for ruta in rutas:
        inicio = time.perf_counter()
        print(f"{ruta.name} ...", end=" ", flush=True)
        ejecutar(ruta, argumentos.tiempo_maximo)
        print(f"{time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
