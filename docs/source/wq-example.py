##################################################################
# Example of Coffea with the Work Queue executor.
#
# To execute, start this application, and then start workers that
# will connect to it and execute tasks.
#
# Note that, as written, this only processes 4 data chunks and
# should complete in a short time.  For a real run,
# change maxchunks=None in the main program below.
#
# For simple testing, you can run one worker manually:
#    work_queue_worker -N coffea-wq-${USER}
#
# Then to scale up, submit lots of workers to your favorite batch system:
#    condor_submit_workers -N coffea-wq-${USER} 32
#
##################################################################

###############################################################
# Sample processor class given in the Coffea manual.
###############################################################

import uproot
from coffea.nanoevents import NanoEventsFactory, BaseSchema

# https://github.com/scikit-hep/uproot4/issues/122
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

import awkward as ak
from coffea import hist, processor

# register our candidate behaviors
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)

class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            "sumw": processor.defaultdict_accumulator(float),
            "mass": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("mass", "$m_{\mu\mu}$ [GeV]", 60, 60, 120),
            ),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        # Note: This is required to ensure that behaviors are registered
        # when running this code in a remote task.        
        ak.behavior.update(candidate.behavior)

        output = self.accumulator.identity()

        dataset = events.metadata['dataset']
        muons = ak.zip({
            "pt": events.Muon_pt,
            "eta": events.Muon_eta,
            "phi": events.Muon_phi,
            "mass": events.Muon_mass,
            "charge": events.Muon_charge,
        }, with_name="PtEtaPhiMCandidate")

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


###############################################################
# Collect and display setup info.
###############################################################

print("------------------------------------------------")
print("Example Coffea Analysis with Work Queue Executor")
print("------------------------------------------------")

import shutil
import getpass
import os.path

wq_env_tarball="coffea-env.tar.gz"
wq_wrapper_path=shutil.which('python_package_run')
wq_master_name="coffea-wq-{}".format(getpass.getuser())

print("Master Name: -N "+wq_master_name)
print("Environment: "+wq_env_tarball)
print("Wrapper Path: "+wq_wrapper_path)

print("------------------------------------------------")


###############################################################
# Sample data sources come from CERN opendata.
###############################################################

fileset = {
    'DoubleMuon': [
        'root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root',
        'root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012C_DoubleMuParked.root',
    ],
}

###############################################################
# Configuration of the Work Queue Executor
###############################################################

work_queue_executor_args = {

    # Options are common to all executors:
    'compression': 1,
    'schema' : BaseSchema,
    'skipbadfiles': False,      # Note that maxchunks only works if this is false.
 
    # Options specific to Work Queue:

    # Additional files needed by the processor, such as local code libraries.
    # 'extra-input-files' : [ 'myproc.py', 'config.dat' ],

    # Resources to allocate per task.
    'resources-mode' : 'auto',  # Adapt task resources to what's observed.
    'resource-monitor': True,   # Measure actual resource consumption

    # With resources set to auto, these are the max values for any task.
    'cores': 2,                  # Cores needed per task.
    'disk': 2000,                # Disk needed per task (MB)
    'memory': 2000,              # Memory needed per task (MB)
    'gpus' : 0,                  # GPUs needed per task.

    # Options to control how workers find this master.
    'master-name': wq_master_name,
    'port': 9123,     # Port for manager to listen on: if zero, will choose automatically.

    # Options to control how the environment is constructed.
    # The named tarball will be transferred to each worker
    # and activated using the wrapper script.
    'environment-file': wq_env_tarball,
    'wrapper' : wq_wrapper_path,

    # Debugging: Display output of task if not empty.
    'print-stdout': True,

    # Debugging: Display notes about each task submitted/complete.
    'verbose': False,

    # Debugging: Produce a lot at the master side of things.
    'debug-log' : 'coffea-wq.log',
}

###############################################################
# Run the analysis via run_uproot_job.
###############################################################

import time
tstart = time.time()

output = processor.run_uproot_job(
    fileset,
    treename='Events',
    processor_instance=MyProcessor(),
    executor=processor.work_queue_executor,
    executor_args=work_queue_executor_args,
    chunksize=100000,

    # Change this to None for a large run:
    maxchunks=4,
)

elapsed = time.time() - tstart

print(output)

