"""Victron VE.Direct battery monitor for the netmon dashboard.

Runs a background thread that reads from the serial port and exposes the latest
state via a thread-safe accessor. If the serial port is unavailable the monitor
silently disables itself.
"""

import threading
import time

try:
    import serial
except ImportError:
    serial = None


class Vedirect:
    (HEX, WAIT_HEADER, IN_KEY, IN_VALUE, IN_CHECKSUM) = range(5)

    def __init__(self, port, baud=19200, timeout=None):
        if serial is None:
            raise ImportError("pyserial not installed")
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self._reset()

    def _reset(self):
        self.header1 = ord("\r")
        self.header2 = ord("\n")
        self.hexmarker = ord(":")
        self.delimiter = ord("\t")
        self.key = ""
        self.value = ""
        self.bytes_sum = 0
        self.state = self.WAIT_HEADER
        self.dict = {}

    def _input(self, byte):
        if byte == self.hexmarker and self.state != self.IN_CHECKSUM:
            self.state = self.HEX
        if self.state == self.WAIT_HEADER:
            self.bytes_sum += byte
            if byte == self.header1:
                self.state = self.WAIT_HEADER
            elif byte == self.header2:
                self.state = self.IN_KEY
            return None
        elif self.state == self.IN_KEY:
            self.bytes_sum += byte
            if byte == self.delimiter:
                self.state = self.IN_CHECKSUM if self.key == "Checksum" else self.IN_VALUE
            else:
                self.key += chr(byte)
            return None
        elif self.state == self.IN_VALUE:
            self.bytes_sum += byte
            if byte == self.header1:
                self.state = self.WAIT_HEADER
                self.dict[self.key] = self.value
                self.key = ""
                self.value = ""
            else:
                self.value += chr(byte)
            return None
        elif self.state == self.IN_CHECKSUM:
            self.bytes_sum += byte
            self.key = ""
            self.value = ""
            self.state = self.WAIT_HEADER
            if self.bytes_sum % 256 == 0:
                self.bytes_sum = 0
                return self.dict
            else:
                self.bytes_sum = 0
        elif self.state == self.HEX:
            self.bytes_sum = 0
            if byte == self.header2:
                self.state = self.WAIT_HEADER
        return None

    def read_data(self):
        while True:
            data = self.ser.read()
            for single_byte in data:
                packet = self._input(single_byte)
                if packet is not None:
                    return packet


class BatteryMonitor:
    def __init__(self, port="/dev/ttyUSB0", v_max_charging=54.0, v_max=52.6, v_min=45.0):
        self.port = port
        self.v_max_charging = v_max_charging
        self.v_max = v_max
        self.v_min = v_min
        self.percentage = 0.0
        self.voltage = 0.0
        self.current = 0.0
        self.state = "Unknown"
        self.available = False
        self.error = ""
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _calc_percentage(self, charger, voltage):
        v_max = self.v_max_charging if charger else self.v_max
        span = v_max - self.v_min
        if span <= 0:
            return 0.0
        return max(0.0, min(100.0, ((voltage - self.v_min) * 100) / span))

    def _loop(self):
        while True:
            try:
                ved = Vedirect(self.port)
                self.available = True
                self.error = ""
                while True:
                    data = ved.read_data()
                    voltage = int(data.get("V", 0)) / 1000.0
                    current = int(data.get("I", 0)) / 1000.0
                    cs = data.get("CS", "0")

                    if cs == "5":
                        pct, state = 100.0, "Charged"
                        charging = False
                    elif cs == "0":
                        pct = self._calc_percentage(charger=False, voltage=voltage)
                        state, charging = "Discharging", False
                    else:
                        pct = self._calc_percentage(charger=True, voltage=voltage)
                        state, charging = "Charging", True

                    pct = max(0.0, min(100.0, pct))
                    with self._lock:
                        self.percentage = pct
                        self.voltage = voltage
                        self.current = current
                        self.state = state
            except Exception as e:
                self.available = False
                self.error = str(e)
                time.sleep(3)

    def snapshot(self):
        with self._lock:
            return {
                "available": self.available,
                "error": self.error,
                "percentage": round(self.percentage, 1),
                "voltage": round(self.voltage, 1),
                "current": round(self.current, 2),
                "state": self.state,
            }
