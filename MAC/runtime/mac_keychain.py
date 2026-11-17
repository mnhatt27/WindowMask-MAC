"""Native user Keychain access; secret bytes never appear in shell arguments."""

from __future__ import annotations

import ctypes as c
import sys


class Keychain:
    def __init__(self, service="com.mystic.windowmask.credentials.v1"):
        if sys.platform != "darwin":
            raise OSError("macOS Keychain requires macOS.")
        self.service = service.encode("utf-8")
        self.api = c.CDLL("/System/Library/Frameworks/Security.framework/Security")
        self.api.SecKeychainFindGenericPassword.argtypes = [
            c.c_void_p,
            c.c_uint32,
            c.c_void_p,
            c.c_uint32,
            c.c_void_p,
            c.POINTER(c.c_uint32),
            c.POINTER(c.c_void_p),
            c.POINTER(c.c_void_p),
        ]
        self.api.SecKeychainAddGenericPassword.argtypes = [
            c.c_void_p,
            c.c_uint32,
            c.c_void_p,
            c.c_uint32,
            c.c_void_p,
            c.c_uint32,
            c.c_void_p,
            c.POINTER(c.c_void_p),
        ]
        self.api.SecKeychainItemModifyAttributesAndData.argtypes = [
            c.c_void_p,
            c.c_void_p,
            c.c_uint32,
            c.c_void_p,
        ]
        self.api.SecKeychainItemDelete.argtypes = [c.c_void_p]
        self.api.SecKeychainItemFreeContent.argtypes = [c.c_void_p, c.c_void_p]
        self.cf = c.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        self.cf.CFRelease.argtypes = [c.c_void_p]
        for name in (
            "SecKeychainFindGenericPassword",
            "SecKeychainAddGenericPassword",
            "SecKeychainItemModifyAttributesAndData",
            "SecKeychainItemDelete",
            "SecKeychainItemFreeContent",
        ):
            getattr(self.api, name).restype = c.c_int32

    @staticmethod
    def _check(status):
        if status:
            raise OSError(
                f"Keychain error {status}. Unlock the login keychain and retry."
            )

    def _find(self, account):
        name = account.encode("utf-8")
        size, data, item = c.c_uint32(), c.c_void_p(), c.c_void_p()
        status = self.api.SecKeychainFindGenericPassword(
            None,
            len(self.service),
            self.service,
            len(name),
            name,
            c.byref(size),
            c.byref(data),
            c.byref(item),
        )
        if status == -25300:  # errSecItemNotFound
            return None, None
        self._check(status)
        try:
            return c.string_at(data, size.value), item
        finally:
            self.api.SecKeychainItemFreeContent(None, data)

    def get(self, account):
        value, item = self._find(account)
        if item:
            self.cf.CFRelease(item)
        return value

    def set(self, account, value):
        _, item = self._find(account)
        try:
            if item:
                self._check(
                    self.api.SecKeychainItemModifyAttributesAndData(
                        item, None, len(value), value
                    )
                )
            else:
                name = account.encode("utf-8")
                self._check(
                    self.api.SecKeychainAddGenericPassword(
                        None,
                        len(self.service),
                        self.service,
                        len(name),
                        name,
                        len(value),
                        value,
                        None,
                    )
                )
        finally:
            if item:
                self.cf.CFRelease(item)

    def delete(self, account):
        _, item = self._find(account)
        if item:
            try:
                self._check(self.api.SecKeychainItemDelete(item))
            finally:
                self.cf.CFRelease(item)
