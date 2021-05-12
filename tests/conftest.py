# Copyright (c) Facebook, Inc. and its affiliates.

import os
from collections import namedtuple
from itertools import groupby

from psutil import Process


LEAK_LIMIT = 10 * 1024 * 1024


_proc = Process(os.getpid())
START = "START"
END = "END"
ConsumedRamLogEntry = namedtuple(
    "ConsumedRamLogEntry", ("nodeid", "on", "consumed_ram")
)
consumed_ram_log = []


def get_consumed_ram():
    return _proc.memory_info().rss


def pytest_runtest_setup(item):
    log_entry = ConsumedRamLogEntry(item.nodeid, START, get_consumed_ram())
    consumed_ram_log.append(log_entry)


def pytest_runtest_teardown(item):
    log_entry = ConsumedRamLogEntry(item.nodeid, END, get_consumed_ram())
    consumed_ram_log.append(log_entry)


def pytest_terminal_summary(terminalreporter):
    grouped = groupby(consumed_ram_log, lambda entry: entry.nodeid)
    for nodeid, (start_entry, end_entry) in grouped:
        leaked = end_entry.consumed_ram - start_entry.consumed_ram
        if leaked > LEAK_LIMIT:
            terminalreporter.write(
                "LEAKED {}MB in {}\n".format(leaked / 1024 / 1024, nodeid)
            )
