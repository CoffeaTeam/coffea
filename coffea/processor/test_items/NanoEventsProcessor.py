from coffea import hist, processor


class NanoEventsProcessor(processor.ProcessorABC):
    def __init__(self, columns=[], canaries=[]):
        self._columns = columns
        self._canaries = canaries
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        mass_axis = hist.Bin("mass", r"$m_{\mu\mu}$ [GeV]", 30000, 0.25, 300)
        pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 30000, 0.25, 300)

        self._accumulator = processor.dict_accumulator(
            {
                'mass': hist.Hist("Counts", dataset_axis, mass_axis),
                'pt': hist.Hist("Counts", dataset_axis, pt_axis),
                'cutflow': processor.defaultdict_accumulator(int),
                'worker': processor.set_accumulator(),
            }
        )

    @property
    def columns(self):
        return self._columns

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()

        dataset = events.metadata['dataset']

        dimuon = events.Muon.choose(2)
        dimuon = dimuon.i0 + dimuon.i1

        magickey = events.Muon.array.content.pt.persistentkey
        if magickey in self._canaries:
            try:
                from distributed import get_worker
                worker = get_worker()
                output['worker'].add(worker.name)
            except ValueError:
                pass

        output['pt'].fill(dataset=dataset, pt=events.Muon.pt.flatten())
        output['mass'].fill(dataset=dataset, mass=dimuon.mass.flatten())
        output['cutflow']['%s_pt' % dataset] += sum(events.Muon.counts)
        output['cutflow']['%s_mass' % dataset] += sum(dimuon.counts)

        return output

    def postprocess(self, accumulator):
        return accumulator
