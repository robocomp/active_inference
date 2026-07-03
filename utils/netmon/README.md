# netmon — monitor web de interconexión RoboComp

Lanza los componentes/agentes de una pila RoboComp, los monitoriza (estado, CPU, memoria)
y sirve un **mapa web** de su interconexión: proxies ICE (RPC), publicación/suscripción por
IceStorm, esferas del grafo DSR y ancho de banda por conexión.

Todo con **stdlib** (sin FastAPI/scapy): web con `http.server`, captura con raw socket.

---

## Lanzamiento

Desde la raíz del proyecto (`~/robocomp/components/active_inference`):

```bash
# Solo capa sensorimotor (arranca Webots) — hospeda la web en http://127.0.0.1:8080
python3 subcognitive.py sub.toml

# Añadir la capa cognitiva (agentes + esfera DSR) — en otra terminal, cualquier orden
python3 cognitive.py cognitive.toml
```

Abre **http://127.0.0.1:8080**. Para en cada terminal con `Ctrl+C` (termina sus procesos,
libera el lock del monitor y borra su registro).

### Flags

| Flag | Efecto |
|------|--------|
| `--web-port N` | Puerto de la web (ambos launchers deben usar el mismo; por defecto `8080`) |
| `--no-web` | No hospedar la web (solo tabla de consola + registro `/tmp`) |
| `--no-bw` | Desactivar la captura de ancho de banda |
| `--no-webots` | No autoarrancar Webots (solo `subcognitive_v2`) |

### Ancho de banda por edge (opcional)

Los `KB/s` solo aparecen si el intérprete tiene permiso de captura. **Una vez**:

```bash
sudo setcap cap_net_raw,cap_net_admin+eip $(readlink -f $(which python3))
```

Sin esto el mapa funciona igual (estado, topología, aristas); solo faltan los bytes/s.

---

## Arquitectura

Cada launcher escribe su estado en un registro común de `/tmp`; el primero que arranca
toma el rol de servidor web y lee **todos** los registros. Si muere, otro lo releva.

```
subcognitive_v2.py ─┐                         /tmp/robocomp_netmon/
cognitive_v2.py ────┼─▶ netmon/launcher.py ──▶  agents/subcognitive.json
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
                   ◆ IceStorm          broker de topics
                   ▢ componentes       sensorimotor con dependencias
        (abajo)    ▢ bridge / fuentes  hubs que no consumen nada
```

- **Nodos** — color por estado: `alive` verde · `up (ICE ✗)` amarillo · `stopped` rojo.
  Formas: `box` componente · `ellipse` agente DSR · `dot` esfera DSR · `diamond` broker ·
  `box` punteado = externo (no gestionado, p. ej. `kinovaarm:12333`).
- **Aristas**:
  - **RPC** (línea sólida, `:puerto interfaz`): engrosa y se pone verde con tráfico (`KB/s`).
  - **pub/sub** (punteada morada): `pub`/`sub` + topic, vía IceStorm. Varios topics del
    mismo par se agrupan en una sola arista multilínea.
  - **DSR** (cian, sin flecha): agente ↔ grafo compartido (lectura/escritura).
- **Tooltip** de cada nodo: qué implementa / requiere / publica / suscribe, y CPU/mem/estado.
- Puedes **arrastrar** nodos para recolocar.
- Botón **Vista** (barra superior): cicla **Orgánico por nivel** (por defecto: force-directed
  con cada nodo anclado a su banda de nivel) → **Jerárquico** → **Orgánico libre**.
  **Reorganizar** re-acomoda; **Tabla** muestra/oculta el panel lateral. La física se congela
  tras estabilizar, así que el mapa no se mueve con los refrescos de estado.

### Control de procesos (panel Tabla)

Cada fila de la tabla trae acciones sobre el componente:

- **↻ relanzar** · **⏹ parar** · **📄 logs**.
- Parar/relanzar se ejecutan de forma segura: el monitor **no** mata procesos ajenos; deja un
  comando en `/tmp/robocomp_netmon/commands/` que **reclama y ejecuta el launcher dueño** del
  proceso (el que lo lanzó). Solo se aceptan `stop`/`restart` sobre componentes conocidos.
- **📄 logs** abre un panel inferior con el `tail` de `~/.local/logs/<name>.{err,out}`
  (conmutable stdout/stderr), refrescado cada 1.5 s y con auto-scroll al final.

---

## Configuración

Los launchers leen un TOML con la lista de componentes:

```toml
[[components]]
name     = "bridge"
cwd      = "~/robocomp/components/webots-bridge"
cmd      = "bin/Webots2Robocomp etc/config"
ice_name = "webots2robocomp:tcp -h localhost -p 10006"   # opcional: proxy para el ping de estado
```

- `ice_name` sirve **solo** para el ping de estado del launcher. La **topología del mapa NO
  se deriva de aquí**, sino de los `etc/config` reales de cada componente
  (`Endpoints.*` = implementa, `Proxies.*` = requiere, `*Prefix`/`*Topic` = pub/sub).
- Los **identities ICE van en minúscula** (`lidar3d`, `camera360rgb`, `webots2robocomp`);
  el ping es case-sensitive. El puerto es el del `Endpoints.<Iface>` del servidor.
- Publishers sin endpoint RPC (p. ej. `python_xbox_controller`) van con `ice_name = ""`.
- **Agentes cognitivos**: se agrupan en una esfera `DSR·dN` por su `[Agent] domain` del
  `config.toml`. Varios domains ⇒ varias esferas.

---

## Ficheros

| Archivo | Rol |
|---------|-----|
| `subcognitive_v2.py`, `cognitive_v2.py` | wrappers finos (raíz del proyecto) |
| `netmon/launcher.py` | lanzar + colector + registro + toma del rol de monitor |
| `netmon/registry.py` | registro `/tmp` compartido + lock del monitor |
| `netmon/topology.py` | grafo desde los configs (RPC, pub/sub, esferas DSR) |
| `netmon/bandwidth.py` | captura loopback (raw AF_PACKET) + atribución por conexión |
| `netmon/server.py` | servidor web (http.server) + API JSON |
| `netmon/static/` | frontend vis-network (`index.html`, `app.js`) |

API: `GET /api/topology` (grafo estático) · `GET /api/state` (estado + `edges_bw`, 1×/s).

---

## Problemas frecuentes

- **Todo sale `Up (ICE ✗)` amarillo** → el proceso vive pero el ping ICE falla: revisa que
  el `ice_name` del TOML (identity minúscula + puerto de `Endpoints.<Iface>`) coincida con
  el `etc/config` real del componente.
- **Un componente sale `Stopped`** → murió por su cuenta (mira `~/.local/logs/<name>.err`);
  típico: hardware ausente (cámara, `/dev/input/js0`) o puerto ya ocupado por una instancia vieja.
- **`bw: sin permisos de captura`** → aplica el `setcap` de arriba.
- **La web no aparece** → comprueba que ningún proceso viejo ocupa el puerto; el lock vivo
  está en `/tmp/robocomp_netmon/monitor.lock`.
