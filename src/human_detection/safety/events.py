"""Where alerts go: the log, a JSON-lines file and, optionally, a webhook."""
import json
import logging
import queue
import threading
import time
import urllib.request
from datetime import datetime, timezone

WEBHOOK_ATTEMPTS = 5  # waits of 1, 2, 4 and 8 s between them; the events file is the record either way
log = logging.getLogger(__name__)


class EventLog:
    """Prints alerts, appends them to a JSON-lines file, and optionally POSTs each
    one as JSON to a webhook (in the background, so video never waits on it)."""

    def __init__(self, camera, path=None, webhook=None):
        self.camera = camera
        self.path = path
        self.models = None  # {name: version}, added to every event once set
        self.queue = None
        if webhook:
            self.queue = queue.Queue()
            threading.Thread(target=self._post, args=(webhook,), daemon=True).start()

    def emit(self, event):
        event = {'time': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'camera': self.camera, **event}
        if self.models:
            event['models'] = self.models
        if event['state'] == 'start':
            log.warning('ALERT  %s', event['message'])
        else:
            log.info('CLEAR  %s', event['message'])
        # Queued first, so a file that cannot be written does not also stop the webhook.
        if self.queue:
            self.queue.put(event)
        if self.path:
            try:
                with open(self.path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(event) + '\n')
            except OSError as error:
                log.error('Cannot write the events file: %s', error)

    def _post(self, url):
        while True:
            event = self.queue.get()
            for attempt in range(WEBHOOK_ATTEMPTS):
                try:
                    request = urllib.request.Request(url, json.dumps(event).encode(),
                                                     {'Content-Type': 'application/json'})
                    urllib.request.urlopen(request, timeout=5).close()
                    break
                # Any error, not only network ones: a reply that is not HTTP must not stop every
                # later alert. The URL may hold a token, so it is kept out of the log.
                except Exception as exc:
                    reason = f'{type(exc).__name__}: {str(exc).replace(url, "<webhook>")}'
                    if attempt == WEBHOOK_ATTEMPTS - 1:
                        log.error('Webhook gave up on a %s event after %d attempts: %s',
                                  event['event'], WEBHOOK_ATTEMPTS, reason)
                    else:
                        time.sleep(2 ** attempt)
            self.queue.task_done()

    def close(self, timeout=5):
        """Give queued webhook posts a few seconds to finish."""
        deadline = time.monotonic() + timeout
        while self.queue and self.queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.05)
