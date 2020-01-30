import os
import blosc
from distributed import WorkerPlugin
from zict import Buffer, Func, LRU, File


class ColumnCacheHolder(WorkerPlugin):
    name = 'cache'

    def __init__(self, maxmem=5e8, maxcompressed=2e9, maxdisk=1e10):
        self._maxmem = maxmem
        self._maxcompressed = maxcompressed
        self._maxdisk = maxdisk

    def setup(self, worker):
        self.cache = Buffer(
            fast={},
            slow=Func(
                dump=blosc.pack_array,
                load=blosc.unpack_array,
                d=Buffer(
                    fast={},
                    slow=LRU(
                        n=self._maxdisk,
                        d=File(os.path.join(worker.local_directory, 'cache')),
                        weight=lambda k, v: len(v),
                    ),
                    n=self._maxcompressed,
                    weight=lambda k, v: len(v),
                ),
            ),
            n=self._maxmem,
            weight=lambda k, v: v.nbytes,
        )

    def teardown(self, worker):
        pass
