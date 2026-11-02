"""Owned SSH connection and request-ID delivery; no inference deadline or route retry."""
from queue import Empty, Full, Queue
from threading import Lock, Thread, Timer
import os
import shlex
import subprocess
import time

from explore_a.remote_online.spec import portable_config, specification
from explore_a.remote_online.wire import IdentityError, VERSION, receive, send


class Client:
    def __init__(self, config, flows, teacher, policy, command=None):
        self.policy, self.spec = policy, specification(config, flows, teacher)
        self.hello = {"type": "hello", "version": VERSION, "spec": self.spec,
                      "config": portable_config(config, self.spec),
                      "flows": list(flows), "teacher": teacher,
                      "send_ahead": policy.get("send_ahead", 2)}
        environment = ["env", "XFORMERS_DISABLED=1",
                       "PYTHONPATH=" + policy["project"] + "/Proj/src", policy["python"],
                       "-u", "-m", "explore_a.remote_online.service"]
        self.command = command or ["ssh", "-T", "-o", "BatchMode=yes", "-o", "Compression=no",
                                   "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
                                   "-o", "ServerAliveCountMax=3", policy["host"],
                                   shlex.join(environment)]
        self.lock, self.closed, self.process = Lock(), False, None
        self.messages, self.ready, self.identifier = None, False, 0
        self.connect_lock, self.write_lock = Lock(), Lock()
        self.replies, self.failure = {}, None

    def _send(self, process, metadata, arrays=()):
        with self.write_lock:
            send(process.stdin, metadata, arrays)

    def _fail_waiters(self, error):
        for channel in self.replies.values():
            try:
                channel.put_nowait(error)
            except Full:
                pass

    def abort(self, expected=None):
        with self.lock:
            if expected is not None and self.process is not expected:
                return
            process, self.process, self.ready = self.process, None, False
            self._fail_waiters(OSError("Remote inference connection closed"))
            if process is not None:
                try:
                    process.kill()
                    process.wait(timeout=1)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass

    def _reader(self, process, messages):
        try:
            while True:
                result = receive(process.stdout)
                metadata, _ = result
                with self.lock:
                    if self.process is not process:
                        return
                    if metadata.get("type") in ("ready", "identity_error"):
                        channel = messages
                    else:
                        channel = self.replies.get(metadata.get("id"))
                        if channel is None:
                            raise IdentityError("Unknown remote response request ID")
                    channel.put_nowait(result)
        except Exception as error:
            with self.lock:
                if self.process is process:
                    self.failure = error
                    self._fail_waiters(error)
            try:
                messages.put(error, timeout=.1)
            except Full:
                pass
        finally:
            process.stdout.close()

    def _receive(self, identifier=None, timeout=None):
        with self.lock:
            channel = self.messages if identifier is None else self.replies[identifier]
        try:
            result = channel.get(timeout=timeout)
        except Empty as error:
            raise TimeoutError("Remote service startup failed") from error
        if isinstance(result, Exception):
            raise result
        return result

    def connect(self):
        with self.connect_lock:
            self._connect()

    def _connect(self):
        if self.failure is not None:
            message = 'Remote connection failed; restart the command explicitly'
            raise OSError(message) from self.failure
        if self.ready:
            return
        with self.lock:
            if self.closed:
                raise OSError("Remote inference client is closed")
            self.messages = Queue(1)
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, env=os.environ.copy())
            process = self.process
        Thread(target=self._reader, args=(process, self.messages), daemon=True,
               name="remote-replies").start()
        timer = Timer(self.policy["startup_timeout_s"], self.abort, args=(process,))
        timer.daemon = True
        timer.start()
        try:
            self._send(process, self.hello)
            message, arrays = self._receive(timeout=self.policy["startup_timeout_s"])
            if message.get("type") == "identity_error":
                raise IdentityError(message["message"])
            if arrays or message.get("type") != "ready" or message.get("spec") != self.spec:
                raise IdentityError("Remote handshake/numeric identity mismatch")
            self.ready = True
        except Exception as error:
            self.failure = error
            self.abort(process)
            raise
        finally:
            timer.cancel()

    def infer(self, payload):
        self.connect()
        with self.lock:
            self.identifier += 1
            identifier, process = self.identifier, self.process
            self.replies[identifier] = Queue(1)
        message = {key:payload[key] for key in ("operation", "flow", "native", "key")}
        message.update(type="infer", id=identifier, canvas=list(payload["canvas"]))
        began = time.monotonic()
        try:
            self._send(process, message, payload["images"])
            reply, arrays = self._receive(identifier)
            if reply.get("id") != identifier or reply.get("key") != payload["key"]:
                raise IdentityError("Remote component/input identity differs")
            if reply.get("type") != "result":
                raise RuntimeError(reply.get("message", "Remote inference failed"))
            from explore_a.frontend import kind
            if reply.get("prediction_kind") != kind(self.hello["config"]):
                raise IdentityError("Remote representation differs")
            names = reply["fields"]
            if len(names) != len(arrays) or len(set(names)) != len(names):
                raise IdentityError("Remote component fields differ")
            return dict(zip(names, arrays)), {"remote_compute_s":reply["compute_s"],
                                             "remote_rpc_s":time.monotonic()-began}
        except Exception as error:
            self.failure = error
            self.abort(process)
            raise
        finally:
            with self.lock:
                self.replies.pop(identifier, None)

    def close(self):
        self.closed = True
        self.abort()
