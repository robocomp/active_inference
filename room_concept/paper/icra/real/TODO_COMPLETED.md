# Tour real ICRA — resultados (REAL_ROBOT_TODO.md ejecutado)

Fecha: 2026-09-14 · Robot: shadow-hp · Deadline paper: 2026-09-15 12:00 CEST

## Resumen ejecutivo

Tour real completado en 3 sesiones encadenadas (295 m totales, muy por encima del mínimo de 100 m).
Resultado principal: **ratio repeat/scatter 13.4x** (sesgo sistemático domina sobre ruido en el 98%
de los pares corner-tramo), con el efecto identificado como **atado al punto de vista** (viewpoint-
attached, 9.9x) — el mismo hallazgo que el paper ya reporta a partir de simulación (8.7x / 6.0x en
`datasets/webots_piso/runs/final_0914/`). El dato real **corrobora** el hallazgo simulado; no
contradice nada ya escrito en el borrador del paper.

## 1 · Banco (bench, sin robot)

| Ítem | Resultado |
|---|---|
| `PoseClampVMax`/`PoseClampWMax` | corregidos a los límites reales del controller (0.5 m/s / 0.75 rad/s; estaban en los defaults 0.7/0.8) |
| SVG de referencia | `layouts/apartamento_layout.svg`, confirmado como el que carga `robot_concept` (`scenario=apartamento`) |
| `gn_selftest` | ALL PASS — se encontró y arregló un bug real en `src/CMakeLists.txt` (le faltaba `door_apertures.cpp` en la lista de fuentes del target, introducido el 2026-09-13 y corregido solo en `wall_slam_selftest`, no en `gn_selftest`) |
| `motion_calib --selftest` | ALL PASS |

## 2 · Fase 1 — parked, motores ENABLED (GO/NO-GO)

- **Veredicto: GO.** σ_v = σ_ω = 0 exacto (encoders de tick cuantizados, robot perfectamente parado —
  no es un sensor roto). Cumple trivialmente "dentro de 2× de la referencia".
- Encontrado y corregido en el camino: el log por-ciclo usado inicialmente (`tmp/sdf_localizer/log_*.csv`)
  hace *alias* del stream de odometría real (documentado en el propio `config.toml`). La medida correcta
  necesita `OdomSampleLog = true` en `[Platform.Shadow]` (estaba en `false`; solo `[Platform.P3Bot]` lo
  tenía activado). Corregido y repetido con datos per-muestra válidos (11081 muestras a ~100 Hz).
- Los parámetros de plataforma (`PreintOdomScaleOmega`, `PreintOdomScaleV`) **transfieren sin
  recalibrar** — no hizo falta ejecutar las Fases 2/3 obligatoriamente, pero se hicieron igualmente
  a título confirmatorio.

## 3 · Fase 2 — closing pivot (confirmatoria)

- Dos intentos (4 + 4 vueltas, 51.55 rad totales). Traslación estable (`s_v≈-1.0` en todas las
  ventanas). **Rotación inestable**: `s_omega≈0.003` en ventanas cortas, diverge en ventanas largas
  (-0.83 a -3.98). Reproducido en dos tandas independientes — no es un fallo puntual de ejecución.
- **No se aplicó ningún valor de esta medida** (demasiado inestable, y la Fase 1 ya daba GO).
  Queda documentado como hallazgo pendiente de investigar tras el deadline — posible relación con
  el mismo tipo de problema de canal de rotación visto en otras partes del sistema.
- Fase 3 (recta medida) omitida por decisión del usuario, dado el tiempo disponible.

## 4 · Paso 5 — El tour real

| Sesión | Distancia | Duración | Notas |
|---|---|---|---|
| 1 | 79.90 m | 6.2 min | primera tirada, 3 tramos estacionarios válidos |
| 2 | 187.59 m | 5.2 min | conducción continua, 0 tramos nuevos válidos (sin paradas largas) |
| 3 | 27.61 m | 5.4 min | 4-5 paradas deliberadas en sitios/orientaciones distintas → 3 tramos nuevos |
| **Total** | **295.10 m** | ~17 min | supera el mínimo de 100 m |

Cada sesión requirió reiniciar `room_concept` (el proceso trunca `corner_probe.csv` al arrancar —
comportamiento documentado y ya conocido de una pérdida de datos anterior el mismo día). Se copiaron
los CSV de cada sesión inmediatamente tras pararla (SIGTERM/Ctrl-C, nunca `kill -9`, confirmando el
proceso muerto antes de copiar) y se concatenaron **renumerando el campo `frame`** con un offset por
sesión — el script de análisis agrupa por número de frame como clave única, así que sin el offset las
tres sesiones (todas empezando en frame 0) se habrían mezclado incorrectamente entre sí.

## 5 · Resultado final (295 m, 3 sesiones combinadas, 182 599 filas / 20 168 frames)

```
NIS/dof overall: 0.836  (objetivo 1.0)

Tramos estacionarios válidos: 6  →  46 pares corner-tramo
  repeats (sesgo)   mediana 0.0624 m
  scatters (ruido)  mediana 0.0046 m
  ratio 13.4x — el sesgo supera al ruido en el 98% de los pares

Test de viewpoint (10 esquinas vistas en ≥3 tramos):
  dispersión dentro de un tramo    0.0046 m
  dispersión entre tramos          0.0459 m   (9.9x)
  → atado al PUNTO DE VISTA (se descarta un offset fijo de mapa)
```

Nota metodológica importante: con solo 3 tramos disponibles (sesiones 1+2), el test de viewpoint daba
el resultado opuesto (1.6x, "wall-attached") — un artefacto de muestra insuficiente, ya que los pocos
tramos disponibles entonces no cubrían puntos de vista realmente distintos. Las paradas deliberadas de
la sesión 3 (a propósito en sitios/orientaciones distintas) revelan la conclusión correcta y son las
que corroboran el resultado ya reportado en simulación.

## 6 · Hallazgo colateral: repo desactualizado

El checkout local de `active_inference` en el robot estaba ~17 commits por detrás de `origin/main`
(sin divergencia, solo desactualizado). El pull trajo `analysis/corner_channel/split_and_nis.py`
(no existía en ningún sitio antes), `paper/icra/PREREGISTRATION_REAL_TOUR.md`, una tirada simulada
completa (`datasets/webots_piso/runs/final_0914/`) y actualizaciones del propio `main.tex`/`main.pdf`.

**Pendiente**: los cambios de esta sesión (dead-reckoning en modo estimate, fix de `gn_selftest` en
CMakeLists, ajustes de config) están guardados en `git stash` en el robot, sin reconciliar todavía
contra lo traído del upstream. No bloquea el tour ya completado, pero hay que resolverlo antes de dar
por cerrado el trabajo de código de esta sesión.

## 7 · Ficheros en esta carpeta

- `EXPERIMENT_LOG.md` — bitácora completa, cronológica, de toda la sesión (útil para auditar cualquier número).
- `real_experiment.md` — propuesta de párrafo para el paper con estos resultados.
- `phase1_bench/`, `phase2_pivot/` — logs de las fases de banco/calibración.
- `real_apartment_0914_2006/`, `real_apartment_0914_sesion2/`, `real_apartment_0914_sesion3/` — datos crudos por sesión.
- `real_apartment_COMBINADO/` — `corner_probe.csv` de las 3 sesiones unido (frames renumerados) + resultados del análisis.
- `split_and_nis.py` — copia del script de análisis usado (traído del repo tras el `git pull`).
