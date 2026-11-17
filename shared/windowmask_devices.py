"""Enumerate Windows GPU adapters without loading a compute driver."""

from __future__ import annotations

import ctypes
import os
import uuid


def gpu_adapters() -> list[dict]:
    if os.name != "nt":
        return []

    class Description(ctypes.Structure):
        _fields_ = [
            ("name", ctypes.c_wchar * 128),
            ("vendor", ctypes.c_uint),
            ("device", ctypes.c_uint),
            ("subsystem", ctypes.c_uint),
            ("revision", ctypes.c_uint),
            ("video", ctypes.c_size_t),
            ("system", ctypes.c_size_t),
            ("shared", ctypes.c_size_t),
            ("luid_low", ctypes.c_uint),
            ("luid_high", ctypes.c_int),
            ("flags", ctypes.c_uint),
        ]

    def method(pointer, index, result, *arguments):
        table = ctypes.cast(
            pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
        ).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(table[index])

    factory = ctypes.c_void_p()
    iid = (ctypes.c_ubyte * 16).from_buffer_copy(
        uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le
    )
    create = ctypes.WinDLL("dxgi").CreateDXGIFactory1
    create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    if create(ctypes.byref(iid), ctypes.byref(factory)) < 0:
        return []
    adapters = []
    seen_intel = set()
    try:
        enumerate_adapter = method(
            factory, 12, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
        )
        for index in range(8):
            adapter = ctypes.c_void_p()
            if enumerate_adapter(factory, index, ctypes.byref(adapter)) < 0:
                break
            try:
                description = Description()
                describe = method(
                    adapter, 10, ctypes.c_long, ctypes.POINTER(Description)
                )
                if (
                    describe(adapter, ctypes.byref(description)) >= 0
                    and not description.flags & 2
                ):
                    if description.vendor in (0x8086, 0x10DE, 0x1002, 0x5143):
                        identity = (description.name, description.device)
                        if description.vendor == 0x8086 and identity in seen_intel:
                            continue
                        if description.vendor == 0x8086:
                            seen_intel.add(identity)
                        adapters.append(
                            {
                                "index": index,
                                "name": description.name,
                                "vendor": description.vendor,
                                "memory": description.video,
                            }
                        )
            finally:
                method(adapter, 2, ctypes.c_ulong)(adapter)
    finally:
        method(factory, 2, ctypes.c_ulong)(factory)
    return adapters
