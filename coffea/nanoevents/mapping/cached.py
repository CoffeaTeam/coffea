from collections.abc import Mapping


class CachedMapping(Mapping):
    """A cache-wrapped mapping
    Reads will call into ``cache`` first, and if no key exists,
    the read will fall back to ``base``, saving the reult into ``cache``.
    """

    def __init__(self, cache, base):
        self.cache = cache
        self.base = base
        self.stats = {"hit": 0, "miss": 0}

    def __getitem__(self, key):
        try:
            value = self.cache[key]
            self.stats["hit"] += 1
            return value
        except KeyError:
            value = self.base[key]
            self.cache[key] = value
            self.stats["miss"] += 1
            return value

    def __iter__(self):
        return iter(self.base)

    def __len__(self):
        return len(self.base)
