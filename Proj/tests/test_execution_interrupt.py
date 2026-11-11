"""Cancellation must exit even when CPU reads or owned remote replies are blocked."""
from pathlib import Path
import os
import subprocess
import sys
import textwrap


def child(code):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(Path(__file__).parents[1]/"src"),
                                         env.get("PYTHONPATH", "")))
    return subprocess.Popen([sys.executable, "-u", "-c", textwrap.dedent(code)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_sigint_exits_with_blocked_prefetch_reader():
    process = child("""
        from threading import Event, Timer
        import os
        import signal
        import time
        from explore_a.execution import prefetch, Timings

        started, blocked = Event(), Event()
        def read(index):
            if index:
                started.set()
                blocked.wait()
            return index
        stream = prefetch(range(3), read, 2, 3, Timings("test"))
        assert next(stream) == 0
        assert started.wait(2)
        Timer(.1, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            stream.close()
        print("CLOSED", flush=True)
    """)
    try:
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, stderr
        assert "CLOSED" in stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_remote_pair_close_aborts_blocked_receive_before_worker_cleanup():
    process = child("""
        from threading import Event
        import subprocess
        import sys
        from explore_a.evaluation.transport import RemotePair

        peer = RemotePair([], {})
        peer.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                        stdout=subprocess.PIPE)
        started = Event()
        def infer(*args):
            started.set()
            return peer.process.stdout.read(1)
        peer._infer = infer
        future = peer.submit(None, False, False, [])
        assert started.wait(2)
        try:
            peer.close()
            assert future.result(timeout=2) == b""
            assert peer.process.poll() is not None
            peer.close()
        finally:
            if peer.process.poll() is None:
                peer.process.kill()
                peer.process.wait()
        print("CLOSED", flush=True)
    """)
    try:
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        assert "CLOSED" in stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
