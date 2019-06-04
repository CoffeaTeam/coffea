from __future__ import print_function
import threading


class HistCollectorCallback(object):
    def __init__(self, hists, redraw_fcn, update=2e5):
        """
            hists: a dictionary of histograms that will be kept by reference and updated
                        with the partial results of the striped session
            redraw_fcn: a callable method that will redraw whatever histograms you want
                        to display in real time as the histograms are updated.  It must
                        return a list of figures
            update: the redraw function will only be called after this many events have
                        been collected from the striped workers
        """
        self._hists = hists
        self._redraw = redraw_fcn
        self._update = update
        self._seen = 0
        self._lock = threading.Lock()

    @property
    def lock(self):
        return self._lock

    def on_streams_update(self, nevents, data):
        """
            This method is called by striped
        """
        if "hists" in data:
            for key in data["hists"]:
                self._hists[key] += data["hists"][key]
        self._seen += nevents
        if self._seen > self._update:
            self._seen = 0
            self.update_histograms()

    def update_histograms(self):
        figs = self._redraw()
        for fig in figs:
            fig.canvas.draw()

    def on_exception(self, wid, info):
        print("Worker exception:")
        print(info)
