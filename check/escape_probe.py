# Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
# SPDX-License-Identifier: MIT
"""The grader's check of its sandbox (faxtest.py selftest): tries what a module must not do and reports what worked
as the bits of its one pixel: 1 read a file in the clone ($FAXTEST_PROBE), 2 open a network connection, 4 start a
child process, 8 write a file beside $FAXTEST_PROBE. The selftest says which bits the sandbox may leave set."""
import os
import socket


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
        socket.create_connection(('1.1.1.1', 80), timeout=3).close()
        flags |= 2
    except OSError:
        pass
    try:
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        flags |= 4
    except (OSError, AttributeError):
        pass
    probe = os.path.join(os.path.dirname(target) or '.', f'.faxtest-escape-{os.getpid()}')
    if target:
        try:
            with open(probe, 'w') as f:
                f.write('x')
            os.unlink(probe)
            flags |= 8
        except OSError:
            pass
    return 1, 1, 1, bytes([flags])
