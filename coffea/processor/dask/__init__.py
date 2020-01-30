from distributed import WorkerPlugin


class ColumnCacheHolder(WorkerPlugin):
    def __init__(self, maxmem=1e9, maxdisk=2e9):
        self._maxmem = maxmem
        self._maxdisk = maxdisk

    def setup(self, worker):
        self.cache = {}

    def teardown(self, worker):
        pass
