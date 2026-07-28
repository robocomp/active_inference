# netmon — monitor web de interconexión RoboComp

Lanza los componentes/agentes de una pila RoboComp, los monitoriza (estado, CPU, memoria)
y sirve un **mapa web** de su interconexión: proxies ICE (RPC), publicación/suscripción por
IceStorm, publicaciones DDS (con ancho de banda real medido, no estimado), esferas del grafo
DSR, ancho de banda por conexión y una ventana de vídeo en directo para el stream RTSP/WebRTC
de mediamtx.

Todo con **stdlib** en el lado Python (sin FastAPI/scapy): web con `http.server`, captura con
raw socket. La única pieza no-stdlib es un binario C++ opcional (`dds_stats_bridge/`) para
medir ancho de banda DDS real — ver más abajo.

---

## Lanzamiento

Desde `~/robocomp/components/active_inference/utils`:

```bash
# Solo capa sensorimotor — hospeda la web en http://0.0.0.0:8080 (accesible desde la red)
python3 subcognitive.py sub.toml          # simulación Webots
python3 subcognitive.py sub_real.toml     # hardware real (robot)

# Añadir la capa cognitiva (agentes + esfera DSR) — en otra terminal, cualquier orden
python3 cognitive.py cognitive.toml
```

Abre **http://\<ip-de-la-máquina\>:8080** (o `http://127.0.0.1:8080` en local). Para en cada
terminal con `Ctrl+C` (termina sus procesos, libera el lock del monitor y borra su registro).

### Flags

| Flag | Efecto |
|------|--------|
| `--web-host IP` | Dirección de bind de la web (por defecto `0.0.0.0`, todas las interfaces) |
| `--web-port N` | Puerto de la web (ambos launchers deben usar el mismo; por defecto `8080`) |
| `--no-web` | No hospedar la web (solo tabla de consola + registro `/tmp`) |
| `--no-bw` | Desactivar la captura de ancho de banda ICE (loopback) |
| `--no-webots` | (`subcognitive.py`) No autoarrancar Webots — solo aplica si no hay `[general]` en el TOML, ver abajo |

### Arranque de Webots / rcnode vía config

En lugar de (o además de) los flags de CLI, cualquier TOML de componentes puede llevar una
tabla `[general]` que manda sobre lo que se pase por línea de comandos:

```toml
[general]
start_webots = false   # p.ej. sub_real.toml: hardware real, sin simulador
start_rcnode = true    # arranca rcnode (icebox) si no está ya corriendo
```

Ambos arranques son idempotentes: si el proceso ya está corriendo (Webots, o `icebox` para
rcnode) se reutiliza en vez de relanzar.

### Ancho de banda ICE por edge (opcional)

Los `KB/s` de las aristas RPC solo aparecen si el intérprete tiene permiso de captura. **Una
vez**:

```bash
sudo setcap cap_net_raw,cap_net_admin+eip $(readlink -f $(which python3))
```

Sin esto el mapa funciona igual (estado, topología, aristas); solo faltan los bytes/s de ICE.
El ancho de banda **DDS** no depende de esto — ver la sección dedicada más abajo.

---

## Arquitectura

Cada launcher escribe su estado en un registro común de `/tmp`; el primero que arranca
toma el rol de servidor web y lee **todos** los registros. Si muere, otro lo releva.

```
subcognitive.py ─┐                          /tmp/robocomp_netmon/
cognitive.py ────┼─▶ netmon/launcher.py ──▶   agents/subcognitive.json
                  │      (cada uno              agents/cognitive.json
                  │       actualiza 1×/s)       monitor.lock  ◀─ 1 solo servidor web
                  └─▶ el que tiene monitor.lock sirve la web con la unión de registros
```

- **Registro** (`registry.py`): `agents/<launcher>.json` con `{name, status, cpu, mem, pid,
  cwd, cmd, layer, domain}`. Registros con más de 5 s sin actualizar se ignoran.
- **Lock** (`monitor.lock`): create-exclusive por PID; el host stale (PID muerto o viejo)
  se roba y se releva automáticamente.

---

## La vista web

Layout **jerárquico de abajo arriba**:

```
        (arriba)   ● DSR·dN            esfera del grafo DSR compartido
                   ⬭ agentes           concept agents (cognitive)  — ovalados
                   ◆ IceStorm          broker de topics ICE
                   ⬡ DDS·dN            broker de dominio DDS (hexágono naranja)
                   ▢ componentes       sensorimotor con dependencias
        (abajo)    ▢ bridge / fuentes  hubs que no consumen nada
```

- **Nodos** — color por estado: `alive` verde · `up (ICE ✗)` amarillo · `stopped` rojo.
  Formas: `box` componente · `ellipse` agente DSR · `dot` esfera DSR · `diamond` broker
  IceStorm · `hexagon` broker DDS · `box` punteado = externo (no gestionado, p. ej.
  `kinovaarm:12333`).
- **Aristas**:
  - **RPC** (línea sólida, `:puerto interfaz`): engrosa y se pone verde con tráfico (`KB/s`).
  - **pub/sub ICE** (punteada morada): `pub`/`sub` + topic, vía IceStorm. Varios topics del
    mismo par se agrupan en una sola arista multilínea.
  - **DDS** (punteada naranja): componente → broker `DDS·dN`, uno o varios topics por
    componente (p. ej. `RGBTopic`/`DepthTopic`). Engrosa y se pone verde con tráfico real
    medido — ver la sección DDS.
  - **DSR** (cian, sin flecha): agente ↔ grafo compartido (lectura/escritura).
- **Tooltip** de cada nodo: qué implementa / requiere / publica / suscribe / DDS (dominio +
  topics), y CPU/mem/estado.
- **Botones de capa** (barra superior): `RPC` / `IceStorm` / `DDS` alternan la visibilidad de
  cada grupo de aristas sin tocar el resto del grafo (útil para desatascar un grafo cargado).
- Puedes **arrastrar** cualquier nodo con el ratón; al soltarlo queda libre permanentemente
  (ya no vuelve a su banda de nivel), para poder recolocar manualmente sin pelearte con el
  layout automático.
- Botón **Vista**: cicla **Orgánico por nivel** (por defecto: force-directed con cada nodo
  anclado a su banda de nivel) → **Jerárquico** → **Orgánico libre**. **Reorganizar**
  re-acomoda; **Tabla** muestra/oculta el panel lateral. La física se congela tras
  estabilizar, así que el mapa no se mueve con los refrescos de estado.

### Live view (barra superior)

Botón **🎥 Live view** abre/cierra un panel flotante y redimensionable con un `<iframe>`
apuntando a la URL que escribas en el campo de texto (se recuerda en `localStorage`). Pensado
para la página HTML con reproductor embebido que sirve **mediamtx** en su puerto WebRTC
(`http://<host>:8889/<path>`, p. ej. `.../theta` para Ricoh) — no necesita ninguna librería
extra en el frontend, es solo un iframe genérico, así que también vale con una URL HLS o
cualquier otra página de vídeo. Al cerrar el panel el iframe se pone a `about:blank` para no
seguir decodificando/consumiendo ancho de banda en segundo plano.

### Control de procesos (panel Tabla)

Cada fila de la tabla trae acciones sobre el componente:

- **↻ relanzar** · **⏹ parar** · **📄 logs**.
- Parar/relanzar se ejecutan de forma segura: el monitor **no** mata procesos ajenos; deja un
  comando en `/tmp/robocomp_netmon/commands/` que **reclama y ejecuta el launcher dueño** del
  proceso (el que lo lanzó). Solo se aceptan `stop`/`restart` sobre componentes conocidos.
  Los procesos externos (Webots, rcnode, `dds_stats_bridge`) no son controlables desde la web.
- **📄 logs** abre un panel inferior con el `tail` de `~/.local/logs/<name>.{err,out}`
  (conmutable stdout/stderr), refrescado cada 1.5 s y con auto-scroll al final.

---

## Ancho de banda DDS real (`dds_stats_bridge`)

Los componentes `*_dds` (p. ej. `ricoh_omni_dds`, `lidar3d_dds`) publican con
`SharedMemoryOnly = true`: los datos van por memoria compartida entre procesos del mismo host,
**nunca por la red** — el sniffing de `bandwidth.py` (raw socket en `lo`) es estructuralmente
incapaz de verlos. Para medir ese tráfico se usa el **módulo de estadísticas nativo de Fast
DDS**, que reporta el throughput real por *DataWriter* independientemente del transporte.

### Cómo funciona

1. `netmon/launcher.py` detecta, a partir del `[DDS]` de la config de cada componente (parseado
   por `topology.component_dds`), cuáles publican por DDS y con qué dominio. A esos procesos les
   inyecta la variable de entorno `FASTDDS_STATISTICS=_fastdds_statistics_publication_throughput`
   al lanzarlos (y al relanzarlos desde la web) — sin tocar su código fuente. Esto hace que cada
   uno publique, además de sus datos normales, su propio throughput (bytes/s instantáneos) en un
   topic interno reservado de Fast DDS.
2. Por cada dominio DDS detectado entre los componentes, el launcher arranca (de forma
   idempotente, igual que Webots/rcnode) una instancia de `netmon/dds_stats_bridge/build/dds_stats_bridge --domain N --out /tmp/robocomp_netmon/dds_stats_dN.json`.
   Este binario C++ standalone se suscribe al topic de estadísticas, resuelve
   GUID-de-writer → nombre de topic real vía discovery normal de DDS, y vuelca a JSON cada 1 s
   `{"<topic>": bytes_por_segundo, ...}` (escritura atómica).
3. `netmon/server.py` fusiona todos los `dds_stats_d*.json` que encuentre y los expone como
   `dds_bw` en `GET /api/state`.
4. `app.js` suma el `dds_bw` de los topics de cada arista DDS del grafo y la anima igual que
   las aristas RPC (grosor + verde con tráfico).

### Construir el binario (una vez, en la máquina donde se vaya a lanzar)

```bash
cd netmon/dds_stats_bridge
mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

Requiere Fast DDS instalado (`fastdds`/`fastcdr` vía `find_package`) **y** un checkout del
código fuente de Fast-DDS que coincida con la versión instalada, porque el código generado a
partir de `fastdds/statistics/types.idl` no se distribuye en los headers públicos — se toma
directamente de `<checkout>/src/cpp/statistics/types/`. La ruta se configura con la variable
CMake `FASTDDS_SRC_TREE` (por defecto `$HOME/software/Fast-DDS`).

Si el binario no existe en `dds_stats_bridge/build/`, el launcher lo detecta, avisa por consola
(`dds_stats_bridge binary missing...`) y sigue funcionando con normalidad — simplemente sin
ancho de banda DDS (el resto del mapa no se ve afectado).

### Notas / limitaciones

- Un componente sin `[DDS] Domain = N` en su config simplemente no participa (no se le inyecta
  la variable de entorno ni cuenta para decidir qué dominios necesitan bridge).
- Un componente puede tener varios topics DDS (`RGBTopic`/`DepthTopic`, etc.) — cualquier clave
  de `[DDS]` que termine en `Topic` cuenta.
- No hay throughput hasta que el *DataWriter* real escribe al menos una muestra: si el
  componente aún no está publicando (p. ej. esperando a que conecte la cámara física), su topic
  simplemente no aparece en `dds_bw` — no es un fallo del bridge.
- El valor es **instantáneo** (bytes desde la última muestra de ese writer / tiempo transcurrido),
  no una media suavizada — puede oscilar entre lecturas de 1 s.

---

## Configuración

Los launchers leen un TOML con la lista de componentes:

```toml
[general]
start_webots = false
start_rcnode = true

[[components]]
name     = "bridge"
cwd      = "~/robocomp/components/webots-bridge"
cmd      = "bin/Webots2Robocomp etc/config"
ice_name = "webots2robocomp:tcp -h localhost -p 10006"   # opcional: proxy para el ping de estado
```

- `ice_name` sirve **solo** para el ping de estado del launcher. La **topología del mapa NO
  se deriva de aquí**, sino de los `etc/config` reales de cada componente
  (`Endpoints.*` = implementa, `Proxies.*` = requiere, `*Prefix`/`*Topic` = pub/sub ICE,
  `[DDS] Domain`/`*Topic` = pub DDS).
- Los **identities ICE van en minúscula** (`lidar3d`, `camera360rgb`, `webots2robocomp`);
  el ping es case-sensitive. El puerto es el del `Endpoints.<Iface>` del servidor.
- Publishers sin endpoint RPC (p. ej. `python_xbox_controller`) van con `ice_name = ""`.
- **Agentes cognitivos**: se agrupan en una esfera `DSR·dN` por su `[Agent] domain` del
  `config.toml`. Varios domains ⇒ varias esferas.

---

## Ficheros

| Archivo | Rol |
|---------|-----|
| `subcognitive.py`, `cognitive.py` | wrappers finos (raíz del proyecto `utils/`) |
| `netmon/launcher.py` | lanzar + colector + registro + toma del rol de monitor + Webots/rcnode/dds_stats_bridge |
| `netmon/registry.py` | registro `/tmp` compartido + lock del monitor |
| `netmon/topology.py` | grafo desde los configs (RPC, pub/sub ICE, DDS, esferas DSR) |
| `netmon/bandwidth.py` | captura loopback ICE (raw AF_PACKET) + atribución por conexión |
| `netmon/server.py` | servidor web (http.server) + API JSON (incluye `dds_bw`) |
| `netmon/static/` | frontend vis-network (`index.html`, `app.js`) |
| `netmon/dds_stats_bridge/` | binario C++ standalone: estadísticas reales de Fast DDS → JSON |

API: `GET /api/topology` (grafo estático) · `GET /api/state` (estado + `edges_bw` + `dds_bw`,
1×/s) · `GET /api/logs?name=&stream=&lines=` · `POST /api/action` (`{action, name}`).

---

## Problemas frecuentes

- **Todo sale `Up (ICE ✗)` amarillo** → el proceso vive pero el ping ICE falla: revisa que
  el `ice_name` del TOML (identity minúscula + puerto de `Endpoints.<Iface>`) coincida con
  el `etc/config` real del componente.
- **Un componente sale `Stopped`** → murió por su cuenta (mira `~/.local/logs/<name>.err`);
  típico: hardware ausente (cámara, `/dev/input/js0`) o puerto ya ocupado por una instancia vieja.
- **`bw: sin permisos de captura`** → aplica el `setcap` de arriba (solo afecta a ICE, no a DDS).
- **La web no aparece** → comprueba que ningún proceso viejo ocupa el puerto; el lock vivo
  está en `/tmp/robocomp_netmon/monitor.lock`.
- **No sale ancho de banda en una arista DDS** → o el binario `dds_stats_bridge` no está
  compilado (revisa la consola del launcher al arrancar), o el componente aún no ha publicado
  ninguna muestra real (p. ej. cámara sin conectar), o su config nunca activó DDS de verdad
  (revisa `~/.local/logs/<name>.out` en busca de `PublishDDS not found, using default false` o
  similar — es un problema de la config del propio componente, no del bridge).
