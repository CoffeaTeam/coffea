import work_queue as wq
import dill

import os
import re
import tempfile
import shutil

import math
import numpy
import scipy

from tqdm.auto import tqdm

from .accumulator import (
    accumulate,
)


# The Work Queue object is global b/c we want to
# retain state between runs of the executor, such
# as connections to workers, cached data, etc.
_wq_queue = None


def work_queue_main(items, function, accumulator, compression_wrapper, decompress_fn, **kwargs):
    """Execute using Work Queue

    For valid parameters, see :py:func:`work_queue_executor` in :py:mod:`executor`.
    For more information, see :ref:`intro-coffea-wq`
    """

    try:
        import work_queue as wq
        import dill
    except ImportError as e:
        print("You must have Work Queue and dill installed to use work_queue_executor!")
        raise e

    global _wq_queue

    # Standard executor options:
    unit = kwargs.pop("unit", "items")
    status = kwargs.pop("status", True)
    desc = kwargs.pop("desc", "Processing")
    clevel = kwargs.pop("compression", 1)
    filepath = kwargs.pop("filepath", ".")

    if clevel is not None:
        function = compression_wrapper(clevel, function)

    # Work Queue specific options:
    verbose_mode = kwargs.pop("verbose", False)
    debug_log = kwargs.pop("debug-log", None)
    stats_log = kwargs.pop("stats-log", None)
    trans_log = kwargs.pop("transactions-log", None)
    extra_input_files = kwargs.pop("extra-input-files", [])
    output = kwargs.pop("print-stdout", False)
    password_file = kwargs.pop("password-file", None)
    env_file = kwargs.pop("environment-file", None)
    wrapper = kwargs.pop("wrapper", shutil.which("python_package_run"))
    resources_mode = kwargs.pop("resources-mode", "fixed")
    cores = kwargs.pop("cores", None)
    memory = kwargs.pop("memory", None)
    disk = kwargs.pop("disk", None)
    gpus = kwargs.pop("gpus", None)
    resource_monitor = kwargs.pop("resource-monitor", False)
    master_name = kwargs.pop("master-name", None)
    port = kwargs.pop("port", None)
    x509_proxy = kwargs.pop("x509_proxy", None)

    dynamic_chunksize = kwargs.pop("dynamic_chunksize", False)
    dynamic_chunksize_targets = kwargs.pop("dynamic_chunksize_targets", {})
    chunksize = kwargs.pop("chunksize", 1024)

    items_total = kwargs["events_total"]

    if port is None:
        if master_name:
            port = 0
        else:
            port = 9123

    if _wq_queue is None:
        _wq_queue = wq.WorkQueue(
            port,
            name=master_name,
            debug_log=debug_log,
            stats_log=stats_log,
            transactions_log=trans_log,
        )

    if env_file and not wrapper:
        raise ValueError(
            "Location of python_package_run could not be determined automatically.\nUse 'wrapper' argument to the work_queue_executor."
        )

    # If explicit resources are given, collect them into default_resources
    default_resources = {}
    if cores:
        default_resources["cores"] = cores
    if memory:
        default_resources["memory"] = memory
    if disk:
        default_resources["disk"] = disk
    if gpus:
        default_resources["gpus"] = gpus

    # if x509_proxy was not provided, then try to get it from env variable or
    # default location.
    if x509_proxy is None:
        if os.environ.get("X509_USER_PROXY", None):
            x509_proxy = os.environ["X509_USER_PROXY"]
        else:
            x509_proxy_default = os.path.join("/tmp", "x509up_u{}".format(os.getuid()))
            if os.path.exists(x509_proxy_default):
                x509_proxy = x509_proxy_default

    # Working within a custom temporary directory:
    with tempfile.TemporaryDirectory(prefix="wq-executor-tmp-", dir=filepath) as tmpdir:

        # Save the executable function in a dilled file.
        with open(os.path.join(tmpdir, "function.p"), "wb") as wf:
            dill.dump(function, wf)

        # Create a wrapper script to run the function.
        command_path = wqex_create_function_wrapper(tmpdir, x509_proxy)

        # Enable monitoring and auto resource consumption, if desired:
        if resource_monitor:
            _wq_queue.enable_monitoring()

        _wq_queue.specify_category_max_resources("default", default_resources)
        if resources_mode == "auto":
            _wq_queue.tune("category-steady-n-tasks", 3)
            _wq_queue.specify_category_max_resources("default", {})
            _wq_queue.specify_category_mode(
                "default", wq.WORK_QUEUE_ALLOCATION_MODE_MAX_THROUGHPUT
            )

        # Make use of the stored password file, if enabled.
        if password_file is not None:
            _wq_queue.specify_password_file(password_file)

        infile_function = os.path.join(tmpdir, "function.p")

        # Keep track of total tasks in each state.
        tasks_submitted = 0
        tasks_done = 0
        items_submitted = 0
        items_done = 0

        print("Listening for work queue workers on port {}...".format(_wq_queue.port))

        # Create a dual progress bar to show submission and completion.
        submit_bar = tqdm(
            total=items_total,
            position=2,
            disable=not status,
            unit=unit,
            desc="Submitted",
        )
        complete_bar = tqdm(
            total=items_total,
            position=1,
            disable=not status,
            unit=unit,
            desc=desc
        )

        # triplets of num of events, wall_time, memory
        task_reports = []

        # ensure items looks like a generator
        if isinstance(items, list):
            items = iter(items)

        # Main loop of executor
        while items_done < items_total or not _wq_queue.empty():
            while (
                items_submitted < items_total and _wq_queue.stats.tasks_waiting < 1
            ):  # and _wq_queue.hungry():
                if items_submitted < 1 or not dynamic_chunksize:
                    item = next(items)
                else:
                    item = items.send(chunksize)
                    chunksize = _compute_chunksize(
                        dynamic_chunksize_targets, chunksize, task_reports
                    )
                    if verbose_mode:
                        print("Updated chunksize:", chunksize)
                task = wqex_create_task(
                    tasks_submitted,
                    item,
                    wrapper,
                    env_file,
                    command_path,
                    infile_function,
                    tmpdir,
                    extra_input_files,
                    x509_proxy,
                )
                task_id = _wq_queue.submit(task)
                tasks_submitted += 1
                items_submitted += len(item)

                if verbose_mode:
                    print("Submitted task (id #{}): {}".format(task_id, task.command))
                submit_bar.update(len(item))
                complete_bar.update(0)

            # When done submitting, look for completed tasks.

            task = _wq_queue.wait(5)
            if task:
                # Display details of the completed task
                wqex_output_task(task, verbose_mode, resource_monitor, output)
                if task.result != 0:
                    # Note that WQ already retries internal failures.
                    # If we get to this point, it's a badly formed task.
                    raise RuntimeError(
                        "Task {} item {} failed with output:\n{}".format(
                            task.id, task.tag, task.output
                        )
                    )

                # The task tag remembers the itemid for us.
                itemid = task.tag
                itemfile = os.path.join(tmpdir, "item_{}.p".format(itemid))
                infile = os.path.join(tmpdir, "item_{}.p".format(itemid))
                outfile = os.path.join(tmpdir, "output_{}.p".format(itemid))

                # Accumulate results from the pickled output
                with open(infile, "rb") as rf:
                    unpickle_input = dill.load(rf)
                    num_items = len(unpickle_input)
                    items_done += num_items
                    # num events, time in seconds, memory used in MB
                    task_reports.append(
                        (
                            num_items,
                            (task.execute_cmd_finish - task.execute_cmd_start) / 1e6,
                            task.resources_measured.memory,
                        )
                    )

                with open(outfile, "rb") as rf:
                    unpickle_output = dill.load(rf)
                    accumulator = accumulate(
                        [
                            unpickle_output
                            if clevel is None
                            else decompress_fn(unpickle_output)
                        ],
                        accumulator,
                    )

                # Remove output files as we go to avoid unbounded disk
                os.remove(itemfile)
                os.remove(outfile)

                tasks_done += 1

                submit_bar.update(0)
                complete_bar.update(num_items)

        submit_bar.close()
        complete_bar.close()

        return accumulator


def wqex_create_function_wrapper(tmpdir, x509_proxy=None):
    """Writes a wrapper script to run dilled python functions and arguments.
    The wrapper takes as arguments the name of three files: function, argument, and output.
    The files function and argument have the dilled function and argument, respectively.
    The file output is created (or overwritten), with the dilled result of the function call.
    The wrapper created is created/deleted according to the lifetime of the work_queue_executor."""

    name = os.path.join(tmpdir, "fn_as_file")

    proxy_basename = ""
    if x509_proxy:
        proxy_basename = os.path.basename(x509_proxy)

    with open(name, mode="w") as f:
        f.write(
            """
#!/usr/bin/env python3
import os
import sys
import dill
import coffea

if "{proxy}":
    os.environ['X509_USER_PROXY'] = "{proxy}"

(fn, arg, out) = sys.argv[1], sys.argv[2], sys.argv[3]

with open(fn, "rb") as f:
    exec_function = dill.load(f)
with open(arg, "rb") as f:
    exec_item = dill.load(f)

pickle_out = exec_function(exec_item)
with open(out, "wb") as f:
    dill.dump(pickle_out, f)

# Force an OS exit here to avoid a bug in xrootd finalization
os._exit(0)
""".format(
                proxy=proxy_basename
            )
        )

    return name


def wqex_create_task(
    itemid,
    item,
    wrapper,
    env_file,
    command_path,
    infile_function,
    tmpdir,
    extra_input_files,
    x509_proxy,
):
    import dill
    from os.path import basename
    import work_queue as wq

    with open(os.path.join(tmpdir, "item_{}.p".format(itemid)), "wb") as wf:
        dill.dump(item, wf)

    infile_item = os.path.join(tmpdir, "item_{}.p".format(itemid))
    outfile = os.path.join(tmpdir, "output_{}.p".format(itemid))

    # Base command just invokes python on the function and data.
    command = "python {} {} {} {}".format(
        basename(command_path),
        basename(infile_function),
        basename(infile_item),
        basename(outfile),
    )

    # If wrapper and env provided, add that.
    if wrapper and env_file:
        command = (
            './{} --environment {} --unpack-to "$WORK_QUEUE_SANDBOX"/{}-env {}'.format(
                basename(wrapper), basename(env_file), basename(env_file), command
            )
        )

    task = wq.Task(command)
    task.specify_category("default")
    task.specify_input_file(command_path, cache=True)
    task.specify_input_file(infile_function, cache=False)
    task.specify_input_file(infile_item, cache=False)

    for f in extra_input_files:
        task.specify_input_file(f, cache=True)

    if x509_proxy:
        task.specify_input_file(x509_proxy, cache=True)

    if wrapper and env_file:
        task.specify_input_file(env_file, cache=True)
        task.specify_input_file(wrapper, cache=True)

    if re.search("://", item.filename) or os.path.isabs(item.filename):
        # This looks like an URL or an absolute path (assuming shared
        # filesystem). Not transfering file.
        pass
    else:
        task.specify_input_file(item.filename, remote_name=item.filename, cache=True)

    task.specify_output_file(outfile, cache=False)

    # Put the item ID into the tag to associate upon completion.
    task.specify_tag("{}".format(itemid))

    return task


def wqex_output_task(task, verbose_mode, resource_mode, output_mode):
    if verbose_mode:
        print(
            "Task (id #{}) complete: {} (return code {})".format(
                task.id, task.command, task.return_status
            )
        )

        print(
            "Allocated cores: {}, memory: {} MB, disk: {} MB, gpus: {}".format(
                task.resources_allocated.cores,
                task.resources_allocated.memory,
                task.resources_allocated.disk,
                task.resources_allocated.gpus,
            )
        )

        if resource_mode:
            print(
                "Measured cores: {}, memory: {} MB, disk {} MB, gpus: {}, runtime {}".format(
                    task.resources_measured.cores,
                    task.resources_measured.memory,
                    task.resources_measured.disk,
                    task.resources_measured.gpus,
                    task.resources_measured.wall_time / 1000000,
                )
            )

    if output_mode and task.output:
        print("Task id #{} output:\n{}".format(task.id, task.output))

    if task.result != 0:
        print("Task id #{} failed with code: {}".format(task.id, task.result))


def _ceil_to_pow2(value):
    return pow(2, math.ceil(math.log2(value)))


def _compute_chunksize(targets, initial_chunksize, task_reports):
    if not targets or len(task_reports) < 1:
        return _ceil_to_pow2(initial_chunksize)

    chunksize_by_time = _compute_chunksize_target(
        targets.get("walltime", 60), [(t, e) for (e, t, mem) in task_reports]
    )
    # chunksize_by_memory = _compute_chunksize_target(targets.get('walltime', 1024), [(mem, e) for (e, t, mem) in task_reports)

    return chunksize_by_time


def _compute_chunksize_target(target, pairs):
    avgs = [e / max(1, target) for (target, e) in pairs]
    quantiles = numpy.quantile(avgs, [0.25, 0.5, 0.75], interpolation="nearest")

    # remove outliers outside the 25%---75% range
    pairs_filtered = []
    for (i, avg) in enumerate(avgs):
        if avg >= quantiles[0] and avg <= quantiles[-1]:
            pairs_filtered.append(pairs[i])

    try:
        # separate into time, numevents arrays
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            [rep[0] for rep in pairs_filtered],
            [rep[1] for rep in pairs_filtered],
        )
    except Exception:
        slope = None

    if (
        slope is None
        or numpy.isnan(slope)
        or numpy.isnan(intercept)
        or slope < 0
        or intercept > 0
    ):
        # we assume that chunksize and walltime have a positive
        # correlation, with a non-negative overhead (-intercept/slope). If
        # this is not true because noisy data, use the avg chunksize/time.
        # slope and intercept may be nan when data falls in a vertical line
        # (specially at the start)
        slope = quantiles[1]
        intercept = 0
    org = (slope * target) + intercept
    exp = math.ceil(math.log2(org))

    # round-up to nearest power of 2, plus minus one power to better sample the space.
    exp += numpy.random.choice([-1, 0, 1])
    pow2 = int(math.pow(2, exp))

    return max(1, pow2)


