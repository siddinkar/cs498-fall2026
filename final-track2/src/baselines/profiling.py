"""Persistent stage wall times for inference and its diagnostic/render outputs."""
from contextlib import contextmanager
import time
import threading
from .pi3x_fusion.common import write_json


class Timings:
    def __init__(self,path):
        self.path=path;self.started=time.perf_counter();self.stages={};self.active=set();self.lock=threading.Lock()
        self.record()

    def record(self):
        write_json(self.path,{'elapsed_seconds':time.perf_counter()-self.started,'stages_seconds':self.stages,'active_stage':', '.join(sorted(self.active)) or None,'active_stages':sorted(self.active)})

    @contextmanager
    def stage(self,name):
        with self.lock:self.active.add(name);self.record()
        start=time.perf_counter()
        try:yield
        finally:
            with self.lock:
                self.stages[name]=time.perf_counter()-start;self.active.remove(name);self.record()
