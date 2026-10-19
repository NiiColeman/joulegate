"""Energy counters and a sampler process that records them.

Every counter's read() returns cumulative joules since the counter was created
(monotone, wrap-safe). The Sampler runs the counters in a separate process so that
polling never holds the measured process's GIL. It reads each counter every
`sample_s` seconds and stores a (time, joules) point only when the value changes, in
shared memory the measured process reads directly. time.perf_counter() is
CLOCK_MONOTONIC on Linux, so both processes share one clock.
"""
import ctypes
import math
import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory

import numpy as np

DEFAULT_SAMPLE_S = {"rapl": 0.0005, "nvml": 0.01, "sysfs": 0.005, "fake": 0.001}


class RaplCounter:
    """Intel RAPL domain (package or DRAM) under /sys/class/powercap."""

    def __init__(self, path):
        self.file = os.path.join(path, "energy_uj")
        with open(os.path.join(path, "max_energy_range_uj")) as f:
            self.wrap = int(f.read())
        self.last, self.total = None, 0

    def read(self):
        with open(self.file) as f:
            raw = int(f.read())
        if self.last is not None:
            d = raw - self.last
            self.total += d if d >= 0 else d + self.wrap
        self.last = raw
        return self.total * 1e-6


class NvmlCounter:
    """NVIDIA total-energy counter through ctypes. With `uuid` (set by build_meter for CUDA runs)
    it opens exactly that GPU. Otherwise `index` counts NVML-visible GPUs, and require_single
    insists only one is visible, since NVML's order need not match CUDA's."""

    def __init__(self, index=0, require_single=True, uuid=None):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self._ok(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        self.h = ctypes.c_void_p()
        if uuid:
            self._ok(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.h)),
                     f"nvmlDeviceGetHandleByUUID({uuid})")
        else:
            count = ctypes.c_uint()
            self._ok(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)), "nvmlDeviceGetCount_v2")
            self.check_single(count.value, require_single)
            self._ok(self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.h)), "nvmlDeviceGetHandleByIndex_v2")
        self.buf, self.base = ctypes.c_ulonglong(), None

    @staticmethod
    def check_single(count, require_single):
        if require_single and count != 1:
            raise RuntimeError(f"NVML sees {count} GPUs, so it is unclear which one the model runs on; "
                               "request exactly one GPU or set require_single: false and index")

    @staticmethod
    def _ok(rc, what):
        if rc != 0:
            raise RuntimeError(f"NVML {what} failed with code {rc}")

    def read(self):
        self._ok(self.lib.nvmlDeviceGetTotalEnergyConsumption(self.h, ctypes.byref(self.buf)),
                 "nvmlDeviceGetTotalEnergyConsumption")
        if self.base is None:
            self.base = self.buf.value
        return (self.buf.value - self.base) * 1e-3


class SysfsPowerCounter:
    """A file reporting instantaneous power (e.g. Jetson INA3221 rail), integrated by trapezoids."""

    def __init__(self, path, unit_w=1e-3):
        self.path, self.unit = path, unit_w
        self.total, self.last_t, self.last_p = 0.0, None, None

    def read(self):
        with open(self.path) as f:
            p = float(f.read().split()[0]) * self.unit
        now = time.perf_counter()
        if self.last_t is not None:
            self.total += 0.5 * (p + self.last_p) * (now - self.last_t)
        self.last_t, self.last_p = now, p
        return self.total


class FakeCounter:
    """Constant power plus `busy_w` per second of CPU time used by the measured process (the
    sampler's parent), optionally quantized like NVML. For tests and dry runs only."""

    def __init__(self, power_w, quantum_j=0.0, busy_w=0.0):
        self.p, self.q, self.busy = power_w, quantum_j, busy_w
        self.stat = f"/proc/{os.getppid()}/stat"
        self.tick = os.sysconf("SC_CLK_TCK")
        self.t0, self.cpu0 = time.perf_counter(), self._cpu()

    def _cpu(self):
        if not self.busy:
            return 0.0
        with open(self.stat) as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / self.tick  # utime + stime, in seconds

    def read(self):
        e = self.p * (time.perf_counter() - self.t0) + self.busy * (self._cpu() - self.cpu0)
        return math.floor(e / self.q) * self.q if self.q else e


def make_counter(spec):
    kind = spec["type"]
    if kind == "rapl":
        return RaplCounter(spec["path"])
    if kind == "nvml":
        return NvmlCounter(spec.get("index", 0), spec.get("require_single", True), spec.get("uuid"))
    if kind == "sysfs":
        return SysfsPowerCounter(spec["path"], spec.get("unit_w", 1e-3))
    if kind == "fake":
        return FakeCounter(spec["power_w"], spec.get("quantum_j", 0.0), spec.get("busy_w", 0.0))
    raise ValueError(f"unknown counter type {kind!r}")


def _sample_s(spec):
    return spec.get("sample_s", DEFAULT_SAMPLE_S[spec["type"]])


def _loop(specs, shm_names, caps, counts, stop, ready, error, msg, cpu):
    shms = []
    try:
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})
        counters = [make_counter(s) for s in specs]
        shms = [shared_memory.SharedMemory(name=n) for n in shm_names]  # the parent owns and unlinks these
        bufs = [np.ndarray((2, c), dtype=np.float64, buffer=s.buf) for c, s in zip(caps, shms)]
        periods = [_sample_s(s) for s in specs]
        due, last = [0.0] * len(specs), [None] * len(specs)
        while not stop.is_set():
            now = time.perf_counter()
            for i, c in enumerate(counters):
                if now < due[i]:
                    continue
                v = c.read()
                t = time.perf_counter()
                if v != last[i]:
                    n = counts[i]
                    if n >= caps[i]:
                        msg.value = b"sample capacity exhausted; raise energy.max_seconds"
                        error.value = 1
                        return
                    bufs[i][0, n], bufs[i][1, n] = t, v
                    counts[i] = n + 1
                    last[i] = v
                due[i] = now + periods[i]
            ready.set()
            time.sleep(max(0.0, min(due) - time.perf_counter()))
    except BaseException as e:  # reported to the parent through `error` and `msg`
        msg.value = f"{type(e).__name__}: {e}".encode()[:511]  # str() keeps the file path, repr() drops it
        error.value = 2
    finally:
        for s in shms:
            s.close()


class Sampler:
    def __init__(self, specs, max_seconds=3600, cpu=None, start_timeout_s=120.0):
        self.names = [s["name"] for s in specs]
        if len(set(self.names)) != len(self.names):
            raise ValueError("counter names must be unique")
        caps = [int(max_seconds / _sample_s(s)) + 16 for s in specs]
        ctx = mp.get_context("spawn")  # a fresh interpreter: safe after CUDA initialisation
        self.shms = [shared_memory.SharedMemory(create=True, size=c * 16) for c in caps]
        self.shm_names = [s.name for s in self.shms]
        # rows (times, joules): each row is contiguous, so np.interp is a binary search, not a copy
        self.bufs = [np.ndarray((2, c), dtype=np.float64, buffer=s.buf) for c, s in zip(caps, self.shms)]
        self.counts = ctx.Array("q", len(specs), lock=False)
        self.stop_ev, self.ready = ctx.Event(), ctx.Event()
        self.error = ctx.Value("i", 0, lock=False)
        self.msg = ctx.Array("c", 512, lock=False)
        self.closed = False
        self.proc = ctx.Process(target=_loop, daemon=True,
                                args=(specs, self.shm_names, caps, self.counts, self.stop_ev, self.ready,
                                      self.error, self.msg, cpu))
        self.proc.start()
        deadline = time.perf_counter() + start_timeout_s  # the child re-imports __main__ (torch, on run.py)
        while not self.ready.is_set():
            if self.error.value or not self.proc.is_alive() or time.perf_counter() > deadline:
                err = self._message()
                self.close()
                raise RuntimeError(f"energy sampler failed to start: {err}")
            time.sleep(0.01)

    def _message(self):
        return self.msg.value.decode(errors="replace") or f"exit code {self.proc.exitcode}"

    def _columns(self, i):
        n = self.counts[i]
        return self.bufs[i][0, :n], self.bufs[i][1, :n]

    def check(self):
        if self.error.value or not self.proc.is_alive():
            raise RuntimeError(f"energy sampler stopped: {self._message()}")

    def snapshot(self):
        out = {}
        for i, name in enumerate(self.names):
            t, v = self._columns(i)
            out[name] = (t.copy(), v.copy())
        return out

    def energy(self, t0, t1):
        out = {}
        for i, name in enumerate(self.names):
            t, v = self._columns(i)
            out[name] = float(np.interp(t1, t, v) - np.interp(t0, t, v))
        return out

    def settled_after(self, t):
        """True once every counter has changed after time t, so energy up to t is final."""
        return all(self.counts[i] and self.bufs[i][0, self.counts[i] - 1] > t for i in range(len(self.names)))

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_ev.set()
        self.proc.join(5)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(5)
        self.bufs = []
        for s in self.shms:
            s.close()
            s.unlink()
