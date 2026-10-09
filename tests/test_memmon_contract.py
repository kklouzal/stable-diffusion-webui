"""shared.mem_mon is a plain query object: extensions (multidiffusion-upscaler's tile scripts) call cuda_mem_get_info()."""

import threading

import torch

from modules import memmon


def test_cuda_mem_get_info_queries_the_monitored_device(monkeypatch):
    queried = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: queried.append(index) or (3, 8))

    monitor = memmon.MemUsageMonitor("MemMon", torch.device("cuda", 1), opts=None)

    assert not monitor.disabled
    assert monitor.cuda_mem_get_info() == (3, 8)
    assert queried == [1, 1]
    assert not isinstance(monitor, threading.Thread)


def test_an_unqueryable_device_disables_the_monitor(monkeypatch):
    def no_device(index):
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(torch.cuda, "mem_get_info", no_device)

    assert memmon.MemUsageMonitor("MemMon", torch.device("cuda", 0), opts=None).disabled
