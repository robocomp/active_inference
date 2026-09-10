# DDS Tuning — media plane (Fast DDS / eProsima)

Host: `robolab@192.168.5.23`
Librería común: `active_inference/common/media_transport/media_transport.cpp` (usada por `ricoh_omni_dds`, `lidar3d_dds` [Helios + Bpearl], `zed_camera`)
Fast DDS: v3.2.2

## 1. Problema original

Transmisión inestable en modo SHM. Diagnóstico:

- `SharedMemTransportDescriptor` se creaba sin fijar `segment_size()` → usaba el valor por defecto de Fast DDS (**512 KB**).
- Los tres `.toml` de los componentes de media plane tienen `DataSharing = false`, así que el **payload completo** (no solo el control RTPS) viaja por ese segmento SHM.
- Tamaño real de las muestras: Image360Frame (`ricoh_omni_dds`) ~5.5 MB; nubes de puntos lidar (`lidar3d_dds`) hasta ~20 MB (`Ice.MessageSizeMax = 20004800`).
- Con `HistoryDepth = 8`, el writer intentaba mantener hasta 8 muestras en cola, agravando la presión sobre un segmento que ni siquiera daba cabida a una.
- Resultado: writer bloqueado/fragmentos descartados en el segmento SHM → cortes de transmisión.

Verificado en vivo: `/dev/shm` tenía ~30 ficheros `fastdds_port*` de 512KB/52KB y decenas de zombies acumulados por reinicios previos, sin presión de espacio total (tmpfs 16G, 1% usado) — el problema no era falta de RAM sino tamaño insuficiente **por segmento**.

## 2. Cambios aplicados

### 2.1 `segment_size` del transporte SHM

`common/media_transport/media_transport.cpp`, función `make_participant()`:

```cpp
auto shm = std::make_shared<eprosima::fastdds::rtps::SharedMemTransportDescriptor>();
shm->segment_size(32 * 1024 * 1024);   // antes: 512KB por defecto (insuficiente)
pqos.transport().user_transports.push_back(shm);
```

Recompilado y enlazado en `lidar3d_dds`, `ricoh_omni_dds` y `zed_camera` (comparten el mismo `.cpp`, un participante compartido por `(domain_id, shm_only)` vía `acquire_participant()`).

### 2.2 `HistoryDepth`: 8 → 2

En los tres `.toml` de producción:

- `hardware/camera/ricoh_omni_dds/etc/config_shadow_real.toml`
- `hardware/laser/lidar3d_dds/etc/config_helios_flip.toml`
- `hardware/laser/lidar3d_dds/etc/config_bpearl_shadow.toml`

```toml
[DDS]
HistoryDepth = 2   # antes 8
```

Motivo: para un media plane en tiempo real, un historial profundo no gana throughput — solo acumula latencia (frames viejos en cola) y, con `data_sharing=false`, presión extra sobre el segmento SHM ya corregido. Depth=2 dá margen de amortiguación sin arrastrar retraso.

### 2.3 Entrega "solo el frame más reciente" en `poll()`

`common/media_transport/media_transport.cpp` — `MediaSubscriber::poll()`, `Image360Subscriber::poll()`, `LidarSubscriber::poll()`.

El bucle `while (reader_->take(...) == RETCODE_OK)` sigue **drenando todo el backlog** (necesario: si no se drena, el pool acotado del reader se llena y el writer se bloquea), pero ahora solo invoca `cb()` **una vez**, con la última muestra válida recibida, en vez de invocarlo por cada muestra del backlog:

```cpp
std::optional<ImageFrame> latest;
std::int64_t latest_recv = 0;
while (reader_->take(data, infos) == efd::RETCODE_OK)
{
    ...
    if (infos[i].valid_data)
    {
        latest = data[i];        // se sobrescribe, solo sobrevive la última
        latest_recv = recv;
        ++delivered;
    }
    ...
}
if (latest)
    cb(*latest, latest_recv);
return delivered;                // sigue contando el total recibido, para stats
```

**`ImuSubscriber::poll()` NO se tocó a propósito.** La IMU necesita cada muestra para integrar (ESKF/dead-reckoning) — descartar muestras intermedias rompería la fusión. El criterio "solo el último dato importa" aplica a vídeo/lidar, no a IMU.

### 2.4 Limpieza de SHM huérfana antes de arrancar

`active_inference/utils/netmon/launcher.py`, nueva función `_check_shm(console)`, invocada en `run_launcher()` justo después de `_remove_existing(components, console)` (mata procesos previos del launcher → limpia lo que esa matanza deja huérfano → arranca los componentes nuevos):

```python
def _check_shm(console):
    before = [f for f in os.listdir("/dev/shm") if f.startswith(("fastdds_", "sem.fastdds_"))]
    if not before:
        return
    result = subprocess.run(["fastdds", "shm", "clean"], capture_output=True, text=True, timeout=15)
    ...
```

Usa `fastdds shm clean` (CLI oficial de Fast DDS 3.2.2 — en esta versión el único subcomando de `shm`; `dump` no existe). Solo borra segmentos SHM sin proceso vivo asociado, nunca toca segmentos en uso.

Aplica a cualquier launcher que use `netmon/launcher.py::run_launcher()`, incluido `subcognitive.py`.

## 3. Configuraciones vitales para tiempo real (verificadas, no tocar sin motivo)

### 3.1 Reliability: RELIABLE, no BEST_EFFORT

`media_transport.cpp`, writer y los 4 readers (Media/Image360/Lidar/Imu):

```cpp
wqos.reliability().kind = std::getenv("MEDIA_BEST_EFFORT")
                              ? efd::BEST_EFFORT_RELIABILITY_QOS
                              : efd::RELIABLE_RELIABILITY_QOS;
```

Comentario original en el código: con BEST_EFFORT el writer no espera y el reader sobrescribe ~2/3 de los frames grandes antes de hacer `take()` → pérdida masiva. RELIABLE + NACK-repair es requisito para no perder frames a 30fps.

`MEDIA_BEST_EFFORT` no está definida en ningún `.bashrc`/`.profile`/`/etc/environment` del host — confirmado que se usa RELIABLE en producción. **No definir esa variable.**

### 3.2 `max_blocking_time` del writer

```cpp
wqos.reliability().max_blocking_time = efd::Duration_t(0, 10 * 1000 * 1000);  // 10 ms
```

Acota el peor caso de bloqueo del productor cuando el historial (`KEEP_LAST`) está lleno de muestras sin ack: un consumidor lento o muerto nunca puede parar al productor más de 10ms. Vital junto con `HistoryDepth` bajo — ambos limitan cuánto puede "atascarse" el pipeline en tiempo real.

### 3.3 Transporte: SHM-only, sin UDP builtin

```cpp
pqos.transport().use_builtin_transports = false;
// solo SharedMemTransportDescriptor
```

Válido porque todos los componentes de media plane corren en el mismo host (`shared_memory_only = true` en los `.toml`, comentario: "SHM transport for same-board"). Si algún día un participante de este dominio corre en otra máquina, esta configuración le impedirá comunicarse (SHM es intra-host) — habría que revisar `shared_memory_only` para ese caso.

### 3.4 `Ice.MessageSizeMax`

En los `.toml` de lidar: `20004800` (~20MB, cubre nubes de puntos completas). En ricoh: `50000000` (50MB). Debe mantenerse ≥ tamaño máximo real del frame; si se aumenta el tamaño de imagen/nube de puntos en el futuro, revisar también `segment_size` (§2.1) — ambos deben crecer juntos.

### 3.5 `DataSharing = false`

Deliberado en los tres `.toml`. Si se activara `data_sharing = true`, el payload se compartiría directamente en el segmento SHM sin copia adicional (cero-copy real), lo que reduciría aún más la presión sobre `segment_size` — no evaluado en esta sesión, posible mejora futura pero requiere revisar el ciclo de vida de los buffers en el lado del publicador (loan/return).

## 4. Pendiente / no verificado en esta sesión

- No se ha reiniciado `ricoh_omni_dds`/`lidar3d_dds` con los binarios recompilados para confirmar en campo que el corte de transmisión SHM desaparece (los procesos se pararon manualmente durante el debugging, no se han vuelto a lanzar todavía).
- No se ha probado `_check_shm()` dentro de un arranque real de `subcognitive.py` (solo se probó la función en aislado, limpiando 5 zombies de 83 ficheros SHM presentes en ese momento).
- No se evaluó `DataSharing = true` como alternativa/complemento a subir `segment_size`.
