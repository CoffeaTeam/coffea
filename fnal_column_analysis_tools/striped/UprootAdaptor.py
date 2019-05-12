from __future__ import division
import time
import uproot
import multiprocessing
import copy

import threading

xrdconfig = {'timeout': None, 'chunkbytes': 1 << 14, 'limitbytes': 1 << 24}


class JaggedStripeDummy(object):
    """
        Such a shame to de-jaggedize and then later re-jaggedize, oh well
    """
    def __init__(self, children):
        self._children = {k: v.content for k, v in children.items()}
        self.count = next(iter(children.values())).counts

    def __repr__(self):
        return "<Jagged object (children=%s) at %08x>" % (",".join(self._children.keys()), id(self))

    def __getattr__(self, key):
        if key in self._children:
            return self._children[key]


class EventsDummy(object):
    def __init__(self, arrays, nevents):
        self._arrays = arrays
        self.nevents = nevents

    def __getattr__(self, key):
        if key in self._arrays:
            return self._arrays[key]


class JobDummy(object):
    def __init__(self, fname, treename, user_params, eventsdef, worker):
        self._fname = fname
        self._treename = treename
        self._user_params = copy.deepcopy(user_params)
        if 'hists' in self._user_params:
            for h in self._user_params['hists'].values():
                h.clear()
        self._eventsdef = eventsdef
        self._worker = worker
        self._out = None

    def __getitem__(self, key):
        return self._user_params[key]

    def send(self, **kwargs):
        if self._out is not None:
            raise Exception("UprootJob only supports one job.send call per worker run call")
        self._out = kwargs

    def __call__(self):
        file = uproot.open(self._fname, xrootdsource=xrdconfig)
        tree = file[self._treename]
        nevents = tree.numentries
        # FIXME: big hack for reading datasets that don't have all columns
        #   up to the worker to handle missing columns, and I'm not sure how striped
        #   will handle this (it doesn't currently)
        bnames = [b for b in self._eventsdef.arraynames() if b in tree]
        arrays = tree.arrays(branches=bnames, namedecode='ascii')
        eventsdummy = self._eventsdef.make_dummy(arrays, nevents)
        self._worker.run(eventsdummy, self)
        return (eventsdummy.nevents, self._out)


class UprootEvents(object):
    def __init__(self, columns, jagged_separator='_'):
        self._flatbranches = [c for c in columns if "." not in c]
        self._jaggedbranches = [c.split(".") for c in columns if "." in c]
        if any(len(b) > 2 for b in self._jaggedbranches):
            raise Exception("Not designed to work for jaggedness larger than 1")
        self._sep = jagged_separator
        self._arraynames = self._flatbranches + [self._sep.join(b) for b in self._jaggedbranches]

    def arraynames(self):
        return self._arraynames

    def make_dummy(self, arrays, nevents):
        parents = set(b[0] for b in self._jaggedbranches)
        for parent in parents:
            children = [self._sep.join(b) for b in self._jaggedbranches if b[0] == parent]
            childnames = [b[1] for b in self._jaggedbranches if b[0] == parent]
            child_dict = {}
            for name, bname in zip(childnames, children):
                child_dict[name] = arrays.pop(bname)
            arrays[parent] = JaggedStripeDummy(child_dict)
        return EventsDummy(arrays, nevents)


class UprootJob(object):
    def __init__(self, dataset, filelist, treename, worker_class, user_callback, user_params):
        self._dataset = dataset
        self._filelist = filelist
        self._treename = treename
        self._worker = worker_class()
        self._user_callback = user_callback
        self._user_params = user_params

        self._eventsdef = UprootEvents(self._worker.Columns)
        self.EventsProcessed = 0
        self._filesProcessed = 0

    def work(self, fname):
        pass

    def run(self, workers=0, progress=True, threaded=True):
        if threaded:
            thread = threading.Thread(target=self._run, args=(workers, progress))
            thread.start()
            return thread
        self._run(workers, progress)

    def _run(self, workers, progress):
        self.TStart = time.time()

        pool = None
        if workers > 0:
            pool = multiprocessing.Pool(processes=workers)

        pbar = None
        if progress:
            from IPython.display import display
            import ipywidgets

            lbl = ipywidgets.Label(self._dataset, layout=ipywidgets.Layout(width="70%"))
            info = ipywidgets.Label("- kevt/s", layout=ipywidgets.Layout(width="10%", display='flex', justify_content='flex-end'))
            pbar = ipywidgets.IntProgress(min=0, max=len(self._filelist), layout=ipywidgets.Layout(width="17%"))
            cancel = ipywidgets.Button(tooltip='Abort processing', icon='times', button_style='danger', layout=ipywidgets.Layout(width="3%"))

            def abort(b):
                if pool is not None:
                    pool.terminate()
                b.disabled = True
            cancel.on_click(abort)
            bar = ipywidgets.HBox([lbl, info, pbar, cancel])
            display(bar)

        try:
            jobs = (JobDummy(fname, self._treename, self._user_params, self._eventsdef, self._worker) for fname in self._filelist)
            if pool is not None:
                res = set(pool.apply_async(job) for job in jobs)
                while True:
                    finished = set()
                    for r in res:
                        if r.ready():
                            if not r.successful():
                                raise r.get()
                            out = r.get()
                            with self._user_callback.lock:
                                self._user_callback.on_streams_update(*out)
                            self.EventsProcessed += out[0]
                            self._filesProcessed += 1
                            finished.add(r)
                    res -= finished
                    finished = None
                    if progress:
                        pbar.value = self._filesProcessed
                        info.value = "{:>5.0f} kevt/s".format(self.EventsProcessed / (time.time() - self.TStart) / 1000)
                        if cancel.disabled:
                            break
                    if len(res) == 0:
                        if progress:
                            cancel.disabled = True
                        break
                    time.sleep(0.2)
            else:
                for job in jobs:
                    out = job()
                    with self._user_callback.lock:
                        self._user_callback.on_streams_update(*out)
                    self.EventsProcessed += out[0]
                    self._filesProcessed += 1
                    if progress:
                        pbar.value = self._filesProcessed
                        info.value = "{:>5.0f} kevt/s".format(self.EventsProcessed / (time.time() - self.TStart) / 1000)
                        if cancel.disabled:
                            break
                if progress:
                    cancel.disabled = True
        except Exception:
            if pool is not None:
                pool.terminate()
            if progress:
                cancel.disabled = True
                info.value = "Exception"
            raise

        pool = None
        self.TFinish = time.time()
