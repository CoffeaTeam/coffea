###############################################################################
# Example of Coffea with the Work Queue executor.
#
# To execute, start this application, and then start workers that will connect
# to it and execute tasks.
#
# Note that, as written, this only processes 4 data chunks and should complete
# in a short time.  For a real run, change maxchunks=None in the main program
# below.
#
# For simple testing this script will automatically use one local worker. To
# scale this up, see the wq.Factory configuration below to change to your
# favorite batch system.
###############################################################################

###############################################################################
# Sample processor class given in the Coffea manual.
###############################################################################
import work_queue as wq

from coffea.processor import Runner
from coffea.processor import WorkQueueExecutor

###############################################################################
# Collect and display setup info.
###############################################################################

print("------------------------------------------------")
print("Example Coffea Analysis with Work Queue Executor")
print("------------------------------------------------")

import getpass

wq_manager_name = "coffea-wq-{}".format(getpass.getuser())
wq_port = 9123

print("Master Name: -M " + wq_manager_name)
print("------------------------------------------------")


###############################################################################
# Define a custom Coffea processor
###############################################################################

from coffea import hist, processor
from coffea.nanoevents.methods import candidate
import awkward as ak

# register our candidate behaviors
ak.behavior.update(candidate.behavior)


class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator(
            {
                "sumw": processor.defaultdict_accumulator(float),
                "mass": hist.Hist(
                    "Events",
                    hist.Cat("dataset", "Dataset"),
                    hist.Bin("mass", r"$m_{\mu\mu}$ [GeV]", 60, 60, 120),
                ),
            }
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        # Note: This is required to ensure that behaviors are registered
        # when running this code in a remote task.
        ak.behavior.update(candidate.behavior)

        output = self.accumulator.identity()

        dataset = events.metadata["dataset"]
        muons = ak.zip(
            {
                "pt": events.Muon_pt,
                "eta": events.Muon_eta,
                "phi": events.Muon_phi,
                "mass": events.Muon_mass,
                "charge": events.Muon_charge,
            },
            with_name="PtEtaPhiMCandidate",
        )

        cut = (ak.num(muons) == 2) & (ak.sum(muons.charge) == 0)
        # add first and second muon in every event together
        dimuon = muons[cut][:, 0] + muons[cut][:, 1]

        output["sumw"][dataset] += len(events)
        output["mass"].fill(
            dataset=dataset,
            mass=dimuon.mass,
        )

        return output

    def postprocess(self, accumulator):
        return accumulator


###############################################################################
# Sample data sources come from CERN opendata.
###############################################################################

fileset = {
    "DoubleMuon": [
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012C_DoubleMuParked.root",
    ],
}


###############################################################################
# Configuration of the Work Queue Executor
###############################################################################

work_queue_executor_args = {
    # Additional files needed by the processor, such as local code libraries.
    # 'extra-input-files' : [ 'myproc.py', 'config.dat' ],
    # Resources to allocate per task.
    "resources_mode": "auto",  # Adapt task resources to what's observed.
    "resource_monitor": True,  # Measure actual resource consumption
    # With resources set to auto, these are the max values for any task.
    "cores": 2,  # Cores needed per task.
    "memory": 500,  # Memory needed per task (MB)
    "disk": 1000,  # Disk needed per task (MB)
    "gpus": 0,  # GPUs needed per task.
    # Options to control how workers find this manager.
    "master_name": wq_manager_name,
    "port": wq_port,  # Port for manager to listen on: if zero, will choose automatically.
    # The named conda environment tarball will be transferred to each worker,
    # and activated. This is useful when coffea is not installed in the remote
    # machines.
    # 'environment_file': wq_env_tarball,
    # Debugging: Display output of task if not empty.
    "print_stdout": False,
    # Debugging: Display notes about each task submitted/complete.
    "verbose": True,
    # Debugging: Produce a lot at the manager side of things.
    "debug_log": "coffea-wq.log",
}

executor = WorkQueueExecutor(**work_queue_executor_args)


###############################################################################
# Run the analysis using local Work Queue workers
###############################################################################

import time

tstart = time.time()

workers = wq.Factory(
    # local runs:
    batch_type="local",
    manager_host_port="localhost:{}".format(wq_port)
    # with a batch system, e.g., condor.
    # (If coffea not at the installation site, then a conda
    # environment_file should be defined in the work_queue_executor_args.)
    # batch_type="condor", manager_name=wq_manager_name
)

workers.max_workers = 2
workers.min_workers = 1
workers.cores = 2
workers.memory = 1000  # MB.
workers.disk = 2000  # MB

with workers:
    # define the Runner instance
    run_fn = Runner(
        executor=executor,
        chunksize=100000,
        maxchunks=4,  # change this to None for a large run
    )
    # execute the analysis on the given dataset
    hists = run_fn(fileset, "Events", MyProcessor())

elapsed = time.time() - tstart


print(hists)
print(hists["mass"])

# (assert only valid when using maxchunks=4)
assert hists["sumw"]["DoubleMuon"] == 400224
