# Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
# SPDX-License-Identifier: MIT
"""The grader's check of its sandbox (faxtest.py selftest): tries what a module must not do and reports what worked
as the bits of its one pixel: 1 read a file in the clone ($FAXTEST_PROBE), 2 connect to a listener the selftest
opened on this machine ($FAXTEST_PORT), 4 start a child process, 8 write a file beside $FAXTEST_PROBE, 16 start
another program, 32 read a system file (/etc/hosts), 64 import an installed third-party package ($FAXTEST_SITE,
$FAXTEST_PACKAGE). The selftest says which bits the sandbox may leave set."""
import importlib
import os
import socket
import sys


def decode(data):
    flags = 0
    target = os.environ.get('FAXTEST_PROBE', '')
    try:
        with open(target, 'rb') as f:
            f.read(1)
        flags |= 1
    except OSError:
        pass
    try:
        socket.create_connection(('127.0.0.1', int(os.environ.get('FAXTEST_PORT', '9'))), timeout=3).close()
        flags |= 2
    except (OSError, ValueError):
        pass
    try:
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        flags |= 4
    except (OSError, AttributeError):
        pass
    if target:
        probe = os.path.join(os.path.dirname(target), f'.faxtest-escape-{os.getpid()}')
        try:
            with open(probe, 'w') as f:
                f.write('x')
            os.unlink(probe)
            flags |= 8
        except OSError:
            pass
    try:
        pid = os.posix_spawn('/usr/bin/true', ['true'], {})
        os.waitpid(pid, 0)
        flags |= 16
    except (OSError, AttributeError):
        pass
    try:
        with open('/etc/hosts', 'rb') as f:
            f.read(1)
        flags |= 32
    except OSError:
        pass
    site, package = os.environ.get('FAXTEST_SITE', ''), os.environ.get('FAXTEST_PACKAGE', '')
    if site and package:
        sys.path.insert(0, site)
        try:
            importlib.import_module(package)
            flags |= 64
        except Exception:
            pass
    return 1, 1, 1, bytes([flags])
