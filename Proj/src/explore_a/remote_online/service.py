"""One GPU executor, bounded receive/write overlap, component-only memory protocol."""
import os
import sys


def serve(fake=False):
    output = os.fdopen(os.dup(1), 'wb', buffering=0)
    os.dup2(2, 1)
    from pathlib import Path
    from queue import Queue
    from threading import Thread
    import time
    import numpy as np
    from explore_a.frontend import kind
    from explore_a.online.components import LocalExecutor, validate
    from explore_a.remote_online.spec import specification
    from explore_a.remote_online.wire import VERSION, receive, send

    hello, arrays = receive(sys.stdin.buffer)
    if arrays or hello.get('version') != VERSION or hello.get('type') != 'hello':
        send(output, {'type':'identity_error', 'message':'Remote component protocol mismatch'})
        return
    config = hello['config']
    project = Path(__file__).resolve().parents[3]
    for name, value in config['weights'].items():
        candidate = (project/value).resolve()
        if not candidate.is_relative_to(project):
            raise ValueError('Remote weight escapes project')
        config['weights'][name] = str(candidate)
    if config.get('paths'):
        config['paths']['dinov2_source'] = str(project/config['paths']['dinov2_source'])
    try:
        spec = hello['spec'] if fake else specification(config, hello['flows'], hello['teacher'])
        if spec != hello['spec']:
            raise ValueError('Remote numeric identity differs')
    except (OSError, ValueError) as error:
        send(output, {'type':'identity_error', 'message':str(error)})
        return
    incoming, outgoing = Queue(hello['send_ahead']), Queue(2)

    def receiver():
        try:
            while True:
                metadata, values = receive(sys.stdin.buffer)
                incoming.put((metadata, values))
        except (OSError, EOFError):
            # The SSH owner is gone; terminate only this private inference service.
            os._exit(0)
        except Exception as error:
            print("Invalid inference stream:", error, file=sys.stderr, flush=True)
            os._exit(1)

    def writer():
        try:
            while (item := outgoing.get()) is not None:
                send(output, *item)
        except (OSError, BrokenPipeError):
            os._exit(1)

    executor = None if fake else LocalExecutor(config, 'cuda')
    send(output, {'type':'ready', 'spec':spec})
    Thread(target=receiver, daemon=True).start()
    writing = Thread(target=writer, daemon=True)
    writing.start()
    try:
        while (item := incoming.get()) is not None:
            message, images = item
            began = time.monotonic()
            try:
                operation, canvas = message['operation'], message['canvas']
                native, flow = message['native'], message['flow']
                expected = {'prediction':1, 'flow':2, 'teacher':3}[operation]
                if len(images) != expected or any(
                        im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3
                        for im in images[:min(expected, 2)]):
                    raise ValueError('Invalid RGB-only component request')
                if operation != 'prediction' and flow not in hello['flows']:
                    raise ValueError('Unexpected flow backend')
                if operation == 'teacher' and not hello['teacher']:
                    raise ValueError('Unexpected MFTIQ request')
                if canvas not in list(spec['data'].values()):
                    raise ValueError('Unknown canvas')
                if fake:
                    if operation == 'prediction':
                        shape = images[0].shape[:2] if native else (1, *canvas)
                        values = {'prediction':np.full(shape, 2, np.float32)}
                    elif operation == 'flow':
                        values = {k:np.zeros((2, *canvas), np.float32)
                                  for k in ('forward', 'backward')}
                    else:
                        values = {'visibility':np.ones((1, *canvas), np.float32)}
                else:
                    values = executor.execute(operation, images, canvas, flow=flow, native=native)
                validate(operation, values, canvas, images[0].shape, config, native)
                outgoing.put(({'type':'result', 'id':message['id'], 'key':message['key'],
                               'fields':list(values), 'prediction_kind':kind(config),
                               'compute_s':time.monotonic()-began}, list(values.values())))
            except Exception as error:
                outgoing.put(({'type':'error', 'id':message['id'], 'key':message.get('key'),
                               'message':str(error)}, []))
    finally:
        outgoing.put(None)
        writing.join()
        output.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fake', action='store_true')
    serve(parser.parse_args().fake)
