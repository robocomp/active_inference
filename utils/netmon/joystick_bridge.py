"""Web joystick -> RoboComp JoystickAdapter bridge.

Republishes axis/button data received over HTTP (from netmon's web joystick panel, see
server_v2.py's /api/joystick) onto the SAME `JoystickAdapter` IceStorm topic
python_xbox_controller already publishes to. SVD48VBase subscribes to that topic and
owns ALL the actual safety logic (see its JoystickAdapter_sendData): the
"joystick_control" button toggles an armed flag that starts False, axes are silently
ignored while disarmed, "stop" disables the motor driver. This bridge does not
reimplement any of that -- it only forwards what the browser asks for, using the exact
same wire message a physical controller would send, plus a deadman watchdog that
force-disarms if the browser goes quiet (lost connection, closed tab) for too long.

Uses Ice.loadSlice() to load JoystickAdapter.ice at runtime (same trick RoboComp's own
generated interfaces.py uses) -- no slice2py build step, no dependency on any other
component's generated/ directory.
"""

import os
import threading
import time

import Ice
import IceStorm

_ICE_DIR = os.path.dirname(__file__)
Ice.loadSlice(f"-I {_ICE_DIR} --all {os.path.join(_ICE_DIR, 'JoystickAdapter.ice')}")
import RoboCompJoystickAdapter as RCJ  # noqa: E402 (must follow loadSlice)

DEADMAN_TIMEOUT = 0.75   # s without a fresh tick() while armed -> auto-disarm
_WATCHDOG_POLL = 0.15
_PULSE_SETTLE = 0.05     # s between a button's step=1 and its step=0 follow-up


class JoystickBridge:
    def __init__(self, topic_manager="IceStorm/TopicManager:default -p 9999"):
        props = Ice.createProperties()
        props.setProperty("Ice.Override.ConnectTimeout", "500")
        init_data = Ice.InitializationData()
        init_data.properties = props
        self._comm = Ice.initialize(init_data)
        self._proxy = None
        self._init_error = None
        try:
            mgr = IceStorm.TopicManagerPrx.checkedCast(self._comm.stringToProxy(topic_manager))
            try:
                topic = mgr.retrieve("JoystickAdapter")
            except IceStorm.NoSuchTopic:
                topic = mgr.create("JoystickAdapter")
            pub = topic.getPublisher().ice_oneway()
            self._proxy = RCJ.JoystickAdapterPrx.uncheckedCast(pub)
        except Ice.Exception as e:
            # rcnode not up, or IceStorm unreachable -- degrade to "no-op bridge" rather
            # than taking the whole web server down; /api/joystick reports the error.
            self._init_error = str(e)

        self.armed = False
        self._lock = threading.Lock()
        self._last_seen = 0.0
        self._stop_event = threading.Event()
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog.start()

    @property
    def available(self):
        return self._proxy is not None

    def _publish(self, axes: dict, buttons: dict):
        if not self._proxy:
            return
        # Plain lists, not RCJ.AxisList/ButtonsList -- those only exist in
        # python_xbox_controller's hand-written interfaces.py (a type-checking
        # convenience it setattr()s onto the module itself), not in what Ice.loadSlice()
        # generates. Ice's Python mapping for a Slice `sequence<T>` is just a list.
        axis_list = [RCJ.AxisParams(name=k, value=float(v)) for k, v in axes.items()]
        button_list = [RCJ.ButtonParams(name=k, step=int(v)) for k, v in buttons.items()]
        self._proxy.sendData(RCJ.TData(id="web_joystick", axes=axis_list, buttons=button_list))

    def _pulse(self, axes: dict, button_name: str):
        """Press-then-release a momentary button (joystick_control/stop/block) as two
        separate messages, regardless of tick cadence -- SVD48VBase only reacts to a
        VALUE CHANGE (see JoystickAdapter_sendData: `if b.step != self.last_buttons[...]`),
        so a bare step=1 that never settles back to 0 would silently no-op the *next*
        press of the same button (1 != 1 is False)."""
        self._publish(axes, {button_name: 1})
        time.sleep(_PULSE_SETTLE)
        self._publish(axes, {button_name: 0})

    def tick(self, axes: dict, arm=None, stop=False, block=False) -> dict:
        """One HTTP request from the browser.
        axes: {"advance": float, "side": float, "rotate": float} -- current stick position.
        arm: True/False to request that transition this tick, None = no change asked.
        stop/block: True to pulse that momentary button this tick.
        Returns {"armed": ..., "available": ...} -- the CLIENT renders our tracked state,
        not an optimistic echo, so a request that arrived while already armed (double
        click, retried request) doesn't make the UI lie about having re-armed anything.
        """
        with self._lock:
            self._last_seen = time.time()
            send_axes = dict(axes) if self.armed else {k: 0.0 for k in axes}
            if arm is True and not self.armed:
                self.armed = True
                self._pulse(send_axes, "joystick_control")
            elif arm is False and self.armed:
                self.armed = False
                self._pulse({k: 0.0 for k in axes}, "joystick_control")
            elif stop or block:
                if stop:
                    self._pulse(send_axes, "stop")
                if block:
                    self._pulse(send_axes, "block")
            else:
                self._publish(send_axes, {})
            return {"armed": self.armed, "available": self.available, "error": self._init_error}

    def _watchdog_loop(self):
        while not self._stop_event.is_set():
            time.sleep(_WATCHDOG_POLL)
            with self._lock:
                if self.armed and (time.time() - self._last_seen) > DEADMAN_TIMEOUT:
                    self.armed = False
                    self._pulse({"advance": 0.0, "side": 0.0, "rotate": 0.0}, "joystick_control")

    def shutdown(self):
        self._stop_event.set()
        if self._comm:
            self._comm.destroy()
