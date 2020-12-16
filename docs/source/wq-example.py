###############################################################
# Sample processor class given in the Coffea manual.
###############################################################

import uproot4
from coffea.nanoevents import NanoEventsFactory, BaseSchema

# https://github.com/scikit-hep/uproot4/issues/122
uproot4.open.defaults["xrootd_handler"] = uproot4.source.xrootd.MultithreadedXRootDSource

import awkward1 as ak
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
# Display some setup info and check common problems.
###############################################################

print("------------------------------------------------")
print("Example Coffea Analysis with Work Queue Executor")
print("------------------------------------------------")

import shutil
import getpass
import os.path

wq_env_tarball="conda-coffea-wq-env.tar.gz"

try:
        wq_wrapper_path=shutil.which('python_package_run')
except:
        print("ERROR: could not find python_package_run in PATH.\nCheck to see that cctools is installed and in the PATH.\n")
        exit(1)

try:
        wq_master_name="coffea-wq-{}".format(getpass.getuser())
except:
        print("ERROR: could not determine current username!")
        exit(1)

if os.path.exists(wq_env_tarball):
	print("Environment tarball: {}".format(wq_env_tarball));
else:
	print("ERROR: environment tarball {} is not present: create it using conda-pack\n",format(wq_env_tarball))
	exit(1)


print("Master Name: -N "+wq_master_name)

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
	'flatten': True,
	'compression': 1,
	'nano' : False,
	'schema' : BaseSchema,
        'skipbadfiles': False,      # Note that maxchunks only works if this is false.
 
	# Options specific to Work Queue: resources to allocate per task.
	'resources-mode' : 'auto',  # Adapt task resources to what's observed.
        'resource-monitor': True,   # Measure actual resource consumption

	# With resources set to auto, these are the max values for any task.
	'cores': 2,                 # Cores needed per task
        'disk': 2000,                # Disk needed per task (MB)
        'memory': 2000,              # Memory needed per task (MB)

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

import time
tstart = time.time()
output = processor.run_uproot_job(
	fileset,
	treename='Events',
	processor_instance=MyProcessor(),
	executor=processor.work_queue_executor,
	executor_args=work_queue_executor_args,
	chunksize=100000,
	maxchunks=None,
)
elapsed = time.time() - tstart
print(output)
