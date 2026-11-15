"""bridge.py -- stream new lines of trace-events.jsonl to the page over a WebSocket.

    pip install websockets
    python bridge.py                      # tails trace-events.jsonl, serves ws://localhost:8765
    python bridge.py --file other.jsonl --port 8765
    python bridge.py --history            # also send what was already in the file

Then open trace-sim/index.html on the same laptop (double-click it, or serve it
locally) and press "Connect live". The hosted copy of the page can't reach the
laptop; it has to be opened locally.

What a page receives: every line appended to the file after the bridge started
(each one a SPEC 3.2 event, sent as the raw JSON text), in order. A page that
connects late first gets everything appended since the bridge started, so it
catches up. With --history it also gets the lines that were in the file before.
Lines that aren't JSON objects are skipped. If the file is truncated or replaced,
the bridge starts again from its beginning.

It only listens on localhost by default: the events say where things are in
someone's home.
"""
import argparse
import asyncio
import json
import os
import sys

try:
    from websockets.asyncio.server import serve       # websockets >= 13
except ImportError:                                     # older releases
    from websockets import serve

POLL = 0.1   # seconds between looks at the file


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Bridge:
    def __init__(self, path, history):
        self.path = path
        self.clients = set()
        self.sent = []            # lines sent since start, replayed to late joiners
        self.pos = 0
        self.ino = None
        self.partial = ""
        if not history and os.path.exists(path):
            self.pos = os.path.getsize(path)
            self.ino = self._id()

    def _id(self):
        st = os.stat(self.path)
        return (st.st_ino, st.st_dev) if st.st_ino else None

    def read_new(self):
        """New complete lines appended to the file since the last look."""
        if not os.path.exists(self.path):
            return []
        size, ident = os.path.getsize(self.path), self._id()
        if size < self.pos or (ident and self.ino and ident != self.ino):
            log("%s was truncated or replaced: reading it from the start" % self.path)
            self.pos, self.partial = 0, ""
        self.ino = ident
        if size == self.pos:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            chunk = f.read()
        self.pos += len(chunk)
        text = self.partial + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self.partial = lines.pop()          # last piece may be a half-written line
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                if not isinstance(json.loads(line), dict):
                    continue
            except json.JSONDecodeError:
                log("skipped a line that isn't JSON: %.60s" % line)
                continue
            out.append(line)
        return out

    async def handler(self, ws, *_):
        self.clients.add(ws)
        log("page connected (%d open); sending %d earlier events" % (len(self.clients), len(self.sent)))
        try:
            for line in list(self.sent):
                await ws.send(line)
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)
            log("page disconnected (%d open)" % len(self.clients))

    async def tail(self):
        while True:
            for line in self.read_new():
                self.sent.append(line)
                for ws in list(self.clients):
                    try:
                        await ws.send(line)
                    except Exception:
                        self.clients.discard(ws)
                log(line)
            await asyncio.sleep(POLL)


async def main():
    ap = argparse.ArgumentParser(description="Serve new lines of trace-events.jsonl on a WebSocket")
    ap.add_argument("--file", default="trace-events.jsonl", help="file to tail (default trace-events.jsonl)")
    ap.add_argument("--host", default="localhost", help="address to listen on (default localhost only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--history", action="store_true", help="also send lines already in the file")
    args = ap.parse_args()
    b = Bridge(args.file, args.history)
    async with serve(b.handler, args.host, args.port):
        log("serving new lines of %s on ws://%s:%d  (Ctrl+C to stop)" % (args.file, args.host, args.port))
        await b.tail()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
