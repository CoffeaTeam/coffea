import dill

import os
import re
import tempfile
import textwrap
import signal

from os.path import basename, join

import math
import numpy
import scipy
import random

from tqdm.auto import tqdm

from .executor import (
    WorkItem,
    _compression_wrapper,
    _decompress,
)

from .accumulator import (
    accumulate,
)


# The Work Queue object is global b/c we want to
# retain state between runs of the executor, such
# as connections to workers, cached data, etc.
_wq_queue = None

# If set to True, workflow stops processing and outputs only the results that
# have been already processed.
early_terminate = False


# This function, that accumulates results from files does not require wq.
# We declare it before checking for wq so that we do not need to install wq at
# the remote site.
def accumulate_result_files(
    chunks_accum_in_mem, clevel, files_to_accumulate, accumulator=None
):
    from coffea.processor import accumulate

    in_memory = []

    # work on local copy of list
    files_to_accumulate = list(files_to_accumulate)
    while files_to_accumulate:
        f = files_to_accumulate.pop()

        # ensure that no files are left unprocessed because lenght of list
        # smaller than desired files in memory.
        chunks_accum_in_mem = min(chunks_accum_in_mem, len(files_to_accumulate))

        with open(f, "rb") as rf:
            result_f = dill.load(rf)
            if clevel is not None:
                result_f = _decompress(result_f)

        if not accumulator:
            accumulator = result_f
            continue

        in_memory.append(result_f)
        if len(in_memory) > chunks_accum_in_mem - 1:
            accumulator = accumulate(in_memory, accumulator)
            while in_memory:
                result = in_memory.pop()  # noqa
                del result
    return accumulator


try:
    from work_queue import WorkQueue, Task
    import work_queue as wq
except ImportError:
    print("work_queue module not available")

    class Task:
        def __init__(self, *args, **kwargs):
            raise ImportError("work_queue not available")

    class WorkQueue:
        def __init__(self, *args, **kwargs):
            raise ImportError("work_queue not available")


class CoffeaWQTask(Task):
    def __init__(
        self, fn_wrapper, infile_function, item_args, itemid, tmpdir, exec_defaults
    ):
        self.itemid = itemid

        self.py_result = ResultUnavailable()
        self._stdout = None

        self.clevel = exec_defaults["compression"]

        self.fn_wrapper = fn_wrapper
        self.infile_function = infile_function

        self.infile_args = join(tmpdir, "args_{}.p".format(self.itemid))
        self.outfile_output = join(tmpdir, "out_{}.p".format(self.itemid))
        self.outfile_stdout = join(tmpdir, "stdout_{}.p".format(self.itemid))

        self.retries = exec_defaults["retries"]

        with open(self.infile_args, "wb") as wf:
            dill.dump(item_args, wf)

        super().__init__(
            self.remote_command(env_file=exec_defaults["environment_file"])
        )

        self.specify_input_file(fn_wrapper, "fn_wrapper", cache=False)
        self.specify_input_file(infile_function, "function.p", cache=False)
        self.specify_input_file(self.infile_args, "args.p", cache=False)
        self.specify_output_file(self.outfile_output, "output.p", cache=False)
        self.specify_output_file(self.outfile_stdout, "stdout.log", cache=False)

        for f in exec_defaults["extra_input_files"]:
            self.specify_input_file(f, cache=True)

        if exec_defaults["x509_proxy"]:
            self.specify_input_file(exec_defaults["x509_proxy"], cache=True)

        if exec_defaults["wrapper"] and exec_defaults["environment_file"]:
            self.specify_input_file(exec_defaults["wrapper"], "py_wrapper", cache=True)
            self.specify_input_file(
                exec_defaults["environment_file"], "env_file", cache=True
            )

    def __len__(self):
        return self.size

    def __str__(self):
        return str(self.itemid)

    def remote_command(self, env_file=None):
        fn_command = "python fn_wrapper function.p args.p output.p >stdout.log 2>&1"
        command = fn_command

        if env_file:
            wrap = (
                './py_wrapper -d -e env_file -u "$WORK_QUEUE_SANDBOX"/{}-env-{} -- {}'
            )
            command = wrap.format(basename(env_file), os.getpid(), fn_command)

        return command

    @property
    def std_output(self):
        if not self._stdout:
            try:
                with open(self.outfile_stdout, "r") as rf:
                    self._stdout = rf.read()
            except Exception:
                self._stdout = None
        return self._stdout

    def _has_result(self):
        return not (
            self.py_result is None or isinstance(self.py_result, ResultUnavailable)
        )

    # use output to return python result, rathern than stdout as regular wq
    @property
    def output(self):
        if not self._has_result():
            try:
                with open(self.outfile_output, "rb") as rf:
                    result = dill.load(rf)
                    if self.clevel is not None:
                        result = _decompress(result)
                    self.py_result = result
            except Exception as e:
                self.py_result = ResultUnavailable(e)
        return self.py_result

    def cleanup_inputs(self):
        os.remove(self.infile_args)

    def cleanup_outputs(self):
        os.remove(self.outfile_output)

    def resubmit(self, tmpdir, exec_defaults):
        if self.retries < 1 or not exec_defaults["split_on_exhaustion"]:
            raise RuntimeError(
                "item {} failed permanently. No more retries left.".format(self.itemid)
            )

        resubmissions = []
        if self.result == wq.WORK_QUEUE_RESULT_RESOURCE_EXHAUSTION:
            _vprint("splitting {} to reduce resource consumption.", self.itemid)
            resubmissions = self.split(tmpdir, exec_defaults)
        else:
            t = self.clone(tmpdir, exec_defaults)
            t.retries = self.retries - 1
            resubmissions = [t]

        for t in resubmissions:
            _vprint(
                "resubmitting {} partly as {} with {} events. {} attempt(s) left.",
                self.itemid,
                t.itemid,
                len(t),
                t.retries,
            )
            _wq_queue.submit(t)

    def clone(self, tmpdir, exec_defaults):
        raise NotImplementedError

    def split(self, tmpdir, exec_defaults):
        raise RuntimeError("task cannot be split any further.")

    def debug_info(self):
        self.output  # load results, if needed

        has_output = "" if self._has_result() else "out"
        msg = "{} with{} result.".format(self.itemid, has_output)
        return msg

    def report(self, output_mode, resource_mode):
        task_failed = (self.result != 0) or (self.return_status != 0)

        if _vprint.verbose_mode or task_failed or output_mode:
            _vprint.printf(
                "{} task id {} item {} with {} events completed on {}. return code {}",
                self.category,
                self.id,
                self.itemid,
                len(self),
                self.hostname,
                self.return_status,
            )

        _vprint(
            "    allocated cores: {}, memory: {} MB, disk: {} MB, gpus: {}",
            self.resources_allocated.cores,
            self.resources_allocated.memory,
            self.resources_allocated.disk,
            self.resources_allocated.gpus,
        )

        if resource_mode:
            _vprint(
                "    measured cores: {}, memory: {} MB, disk {} MB, gpus: {}, runtime {}",
                self.resources_measured.cores,
                self.resources_measured.memory,
                self.resources_measured.disk,
                self.resources_measured.gpus,
                (self.cmd_execution_time) / 1e6,
            )

        if (task_failed or output_mode) and self.std_output:
            _vprint.print("    output:")
            _vprint.print(self.std_output)

        if task_failed:
            # Note that WQ already retries internal failures.
            # If we get to this point, it's a badly formed task
            info = self.debug_info()
            _vprint.printf(
                "task id {} item {} failed: {}\n    {}",
                self.id,
                self.itemid,
                self.result_str,
                info,
            )

        return not task_failed


class PreProcCoffeaWQTask(CoffeaWQTask):
    tasks_counter = 0
    infile_function = None

    def __init__(
        self, fn_wrapper, infile_function, item, tmpdir, exec_defaults, itemid=None
    ):
        PreProcCoffeaWQTask.tasks_counter += 1

        if not itemid:
            itemid = "pre_{}".format(PreProcCoffeaWQTask.tasks_counter)

        self.item = item

        self.size = 1
        super().__init__(
            fn_wrapper, infile_function, [item], itemid, tmpdir, exec_defaults
        )

        self.specify_category("preprocessing")

        if re.search("://", item.filename) or os.path.isabs(item.filename):
            # This looks like an URL or an absolute path (assuming shared
            # filesystem). Not transfering file.
            pass
        else:
            self.specify_input_file(
                item.filename, remote_name=item.filename, cache=True
            )

    def clone(self, tmpdir, exec_defaults):
        return PreProcCoffeaWQTask(
            self.fn_wrapper,
            self.infile_function,
            self.item,
            tmpdir,
            exec_defaults,
            self.itemid,
        )

    def debug_info(self):
        i = self.item
        msg = super().debug_info()
        return "{} {}".format((i.dataset, i.filename, i.treename), msg)


class ProcCoffeaWQTask(CoffeaWQTask):
    tasks_counter = 0

    def __init__(
        self, fn_wrapper, infile_function, item, tmpdir, exec_defaults, itemid=None
    ):
        self.size = len(item)

        ProcCoffeaWQTask.tasks_counter += 1
        if not itemid:
            itemid = "p_{}".format(ProcCoffeaWQTask.tasks_counter)

        self.item = item

        super().__init__(
            fn_wrapper, infile_function, [item], itemid, tmpdir, exec_defaults
        )

        self.specify_category("processing")

        if re.search("://", item.filename) or os.path.isabs(item.filename):
            # This looks like an URL or an absolute path (assuming shared
            # filesystem). Not transfering file.
            pass
        else:
            self.specify_input_file(
                item.filename, remote_name=item.filename, cache=True
            )

    def clone(self, tmpdir, exec_defaults):
        return ProcCoffeaWQTask(
            self.fn_wrapper,
            self.infile_function,
            self.item,
            tmpdir,
            exec_defaults,
            self.itemid,
        )

    def split(self, tmpdir, exec_defaults):
        total = len(self.item)

        if total < 2:
            raise RuntimeError("processing task cannot be split any further.")

        # if the chunksize was updated to be less than total, then use that.
        # Otherwise, just partition the task in two.
        target_chunksize = exec_defaults["updated_chunksize"]
        if total <= target_chunksize:
            target_chunksize = math.ceil(total / 2)

        n = max(math.ceil(total / target_chunksize), 1)
        actual_chunksize = int(math.ceil(total / n))

        splits = []
        start = self.item.entrystart
        while start < self.item.entrystop:
            stop = min(self.item.entrystop, start + actual_chunksize)

            w = WorkItem(
                self.item.dataset,
                self.item.filename,
                self.item.treename,
                start,
                stop,
                self.item.fileuuid,
                self.item.usermeta,
            )

            t = self.__class__(
                self.fn_wrapper, self.infile_function, w, tmpdir, exec_defaults
            )

            start = stop
            splits.append(t)

        return splits

    def debug_info(self):
        i = self.item
        msg = super().debug_info()
        return "{} {}".format(
            (i.dataset, i.filename, i.treename, i.entrystart, i.entrystop), msg
        )


class AccumCoffeaWQTask(CoffeaWQTask):
    tasks_counter = 0

    def __init__(
        self,
        fn_wrapper,
        infile_function,
        tasks_to_accumulate,
        tmpdir,
        exec_defaults,
        itemid=None,
    ):
        AccumCoffeaWQTask.tasks_counter += 1

        if not itemid:
            itemid = "accum_{}".format(AccumCoffeaWQTask.tasks_counter)

        self.tasks_to_accumulate = tasks_to_accumulate
        self.size = sum(len(t) for t in self.tasks_to_accumulate)

        args = [exec_defaults["chunks_accum_in_mem"], exec_defaults["compression"]]
        args = args + [[basename(t.outfile_output) for t in self.tasks_to_accumulate]]

        super().__init__(
            fn_wrapper, infile_function, args, itemid, tmpdir, exec_defaults
        )

        self.specify_category("accumulating")

        for t in self.tasks_to_accumulate:
            self.specify_input_file(t.outfile_output, cache=False)

    def cleanup_inputs(self):
        super().cleanup_inputs()
        # cleanup files associated with results already accumulated
        for t in self.tasks_to_accumulate:
            t.cleanup_outputs()

    def clone(self, tmpdir, exec_defaults):
        return AccumCoffeaWQTask(
            self.fn_wrapper,
            self.infile_function,
            self.tasks_to_accumulate,
            tmpdir,
            exec_defaults,
            self.itemid,
        )

    def debug_info(self):
        tasks = self.tasks_to_accumulate

        msg = super().debug_info()

        results = [
            CoffeaWQTask.debug_info(t)
            for t in tasks
            if isinstance(t, AccumCoffeaWQTask)
        ]
        results += [
            t.debug_info() for t in tasks if not isinstance(t, AccumCoffeaWQTask)
        ]

        return "{} accumulating: [{}] ".format(msg, "\n".join(results))


def work_queue_main(items, function, accumulator, **kwargs):
    """Execute using Work Queue

    For valid parameters, see :py:func:`work_queue_executor` in :py:mod:`executor`.
    For more information, see :ref:`intro-coffea-wq`
    """

    global _wq_queue

    _check_dynamic_chunksize_targets(kwargs["dynamic_chunksize"])

    clevel = kwargs["compression"]
    if clevel is not None:
        function = _compression_wrapper(clevel, function)
        accumulate_fn = _compression_wrapper(clevel, accumulate_result_files)
    else:
        accumulate_fn = accumulate_result_files

    _vprint.verbose_mode = kwargs["verbose"] or kwargs["print_stdout"]
    _vprint.status_mode = kwargs["status"]

    if not kwargs["port"]:
        kwargs["port"] = 0 if kwargs["master_name"] else 9123

    if kwargs["environment_file"] and not kwargs["wrapper"]:
        raise ValueError(
            "Location of python_package_run could not be determined automatically.\nUse 'wrapper' argument to the work_queue_executor."
        )

    if _wq_queue is None:
        _wq_queue = WorkQueue(
            port=kwargs["port"],
            name=kwargs["master_name"],
            debug_log=kwargs["debug_log"],
            stats_log=kwargs["stats_log"],
            transactions_log=kwargs["transactions_log"],
        )

        # Make use of the stored password file, if enabled.
        if kwargs["password_file"] is not None:
            _wq_queue.specify_password_file(kwargs["password_file"])

        print("Listening for work queue workers on port {}...".format(_wq_queue.port))
        # perform a wait to print any warnings before progress bars
        _wq_queue.wait(0)

    _declare_resources(kwargs)

    # Working within a custom temporary directory:
    with tempfile.TemporaryDirectory(
        prefix="wq-executor-tmp-", dir=kwargs["filepath"]
    ) as tmpdir:
        fn_wrapper = _create_fn_wrapper(kwargs["x509_proxy"], tmpdir=tmpdir)
        infile_function = _function_to_file(
            function, prefix_name=kwargs["function_name"], tmpdir=tmpdir
        )
        infile_accum_fn = _function_to_file(
            accumulate_fn, prefix_name="accum", tmpdir=tmpdir
        )

        if kwargs["custom_init"]:
            kwargs["custom_init"](_wq_queue)

        if kwargs["desc"] == "Preprocessing":
            return _work_queue_preprocessing(
                items, accumulator, fn_wrapper, infile_function, tmpdir, kwargs
            )
        else:
            return _work_queue_processing(
                items,
                accumulator,
                fn_wrapper,
                infile_function,
                infile_accum_fn,
                tmpdir,
                kwargs,
            )


def _work_queue_processing(
    items,
    accumulator,
    fn_wrapper,
    infile_function,
    infile_accum_fn,
    tmpdir,
    exec_defaults,
):

    # Keep track of total tasks in each state.
    items_submitted = 0
    items_done = 0

    # triplets of num of events, wall_time, memory
    task_reports = []

    # tasks with results to accumulate, sorted by the number of events
    tasks_to_accumulate = []

    # ensure items looks like a generator
    if isinstance(items, list):
        items = iter(items)

    items_total = exec_defaults["events_total"]

    # "chunksize" is the original chunksize passed to the executor. Always used
    # if dynamic_chunksize is not given.
    chunksize = exec_defaults["chunksize"]

    # keep a record of the latest computed chunksize, if any
    exec_defaults["updated_chunksize"] = exec_defaults["chunksize"]

    progress_bars = _make_progress_bars(exec_defaults)

    signal.signal(signal.SIGINT, _handle_early_terminate)

    # Main loop of executor
    while (not early_terminate and items_done < items_total) or not _wq_queue.empty():
        while (
            items_submitted < items_total and _wq_queue.hungry() and not early_terminate
        ):
            update_chunksize = (
                items_submitted > 0 and exec_defaults["dynamic_chunksize"]
            )
            if update_chunksize:
                _vprint(
                    "current chunksize {}",
                    _compute_chunksize(task_reports, exec_defaults, sample=False),
                )

                chunksize = _compute_chunksize(task_reports, exec_defaults)

            task = _submit_proc_task(
                fn_wrapper,
                infile_function,
                items,
                chunksize,
                update_chunksize,
                tmpdir,
                exec_defaults,
            )
            items_submitted += len(task)
            progress_bars["submit"].update(len(task))

        # When done submitting, look for completed tasks.
        task = _wq_queue.wait(5)

        # refresh progress bars
        for bar in progress_bars.values():
            bar.update(0)

        if task:
            # Evaluate and display details of the completed task
            success = task.report(
                exec_defaults["print_stdout"], exec_defaults["resource_monitor"]
            )

            if not success:
                task.resubmit(tmpdir, exec_defaults)
            else:
                tasks_to_accumulate.append(task)

                if task.category == "processing":
                    items_done += len(task)
                    progress_bars["process"].update(len(task))

                    # gather statistics for dynamic chunksize
                    task_reports.append(
                        (
                            len(task),
                            (task.cmd_execution_time) / 1e6,
                            task.resources_measured.memory,
                        )
                    )
                else:
                    progress_bars["accumulate"].update(1)

                force_last_accum = (items_done >= items_total) or early_terminate
                tasks_to_accumulate = _submit_accum_tasks(
                    fn_wrapper,
                    infile_accum_fn,
                    tasks_to_accumulate,
                    force_last_accum,
                    tmpdir,
                    exec_defaults,
                )
                progress_bars["accumulate"].total = math.ceil(
                    items_total * AccumCoffeaWQTask.tasks_counter / items_done
                )

                # Remove input files as we go to avoid unbounded disk
                # we do not remove outputs, as they are used by further accumulate tasks
                task.cleanup_inputs()

    for bar in progress_bars.values():
        bar.close()

    if items_done < items_total:
        _vprint.printf("\nWARNING: Not all items were processed.\n")
    accumulator = _final_accumulation(
        accumulator, tasks_to_accumulate, exec_defaults["compression"]
    )

    if exec_defaults["dynamic_chunksize"]:
        _vprint(
            "final chunksize {}",
            _compute_chunksize(task_reports, exec_defaults, sample=False),
        )

    return accumulator


def _handle_early_terminate(signum, frame):
    global early_terminate

    if early_terminate:
        raise KeyboardInterrupt
    else:
        _vprint.printf(
            "********************************************************************************"
        )
        _vprint.printf("Canceling processing tasks for final accumulation.")
        _vprint.printf("C-c again to immediately terminate.")
        _vprint.printf(
            "********************************************************************************"
        )
        early_terminate = True
        _wq_queue.cancel_by_category("processing")


def _final_accumulation(accumulator, tasks_to_accumulate, compression):
    if len(tasks_to_accumulate) < 1:
        raise RuntimeError("No results available.")
    elif len(tasks_to_accumulate) > 1:
        _vprint.printf(
            "Not all results ({}) were accumulated in an accumulation job. Accumulating locally.".format(
                len(tasks_to_accumulate)
            )
        )

    _vprint("Performing final accumulation...")

    accumulator = accumulate_result_files(
        2, compression, [t.outfile_output for t in tasks_to_accumulate], accumulator
    )
    for t in tasks_to_accumulate:
        t.cleanup_outputs()
    return accumulator


def _work_queue_preprocessing(
    items, accumulator, fn_wrapper, infile_function, tmpdir, exec_defaults
):
    preprocessing_bar = tqdm(
        desc="Preprocessing",
        total=len(items),
        disable=not exec_defaults["status"],
        unit=exec_defaults["unit"],
        bar_format=exec_defaults["bar_format"],
    )

    for item in items:
        task = PreProcCoffeaWQTask(
            fn_wrapper, infile_function, item, tmpdir, exec_defaults
        )
        _wq_queue.submit(task)
        _vprint("submitted preprocessing task {}", task.id)

    while not _wq_queue.empty():
        task = _wq_queue.wait(5)
        if task:
            success = task.report(
                exec_defaults["print_stdout"], exec_defaults["resource_monitor"]
            )
            if success:
                accumulator = accumulate([task.output], accumulator)
                preprocessing_bar.update(1)
                task.cleanup_inputs()
                task.cleanup_outputs()
            else:
                task.resubmit(tmpdir, exec_defaults)

    preprocessing_bar.close()

    return accumulator


def _declare_resources(exec_defaults):
    # If explicit resources are given, collect them into default_resources
    default_resources = {}
    if exec_defaults["cores"]:
        default_resources["cores"] = exec_defaults["cores"]
    if exec_defaults["memory"]:
        default_resources["memory"] = exec_defaults["memory"]
    if exec_defaults["disk"]:
        default_resources["disk"] = exec_defaults["disk"]
    if exec_defaults["gpus"]:
        default_resources["gpus"] = exec_defaults["gpus"]

    # Enable monitoring and auto resource consumption, if desired:
    _wq_queue.tune("category-steady-n-tasks", 3)

    monitor_enabled = False

    # if resource_monitor is given, and not 'off', then monitoring is activated.
    # anything other than 'measure' is assumed to be 'watchdog' mode, where in
    # addition to measuring resources, tasks are killed if they go over their
    # resources.
    if exec_defaults["resource_monitor"] and exec_defaults["resource_monitor"] != "off":
        monitor_enabled = True
        _wq_queue.enable_monitoring(
            watchdog=(exec_defaults["resource_monitor"] != "measure")
        )

    # activate monitoring as a watchdog if it has not been explicitely
    # activated and we are using an automatic resource allocation.
    if not monitor_enabled:
        if (
            exec_defaults["resources_mode"]
            and exec_defaults["resources_mode"] != "fixed"
        ):
            _wq_queue.enable_monitoring(watchdog=True)

    for category in "default preprocessing processing accumulating".split():
        _wq_queue.specify_category_max_resources(category, default_resources)

        if exec_defaults["resources_mode"] != "fixed":
            _wq_queue.specify_category_mode(category, wq.WORK_QUEUE_ALLOCATION_MODE_MAX)

            if (
                category == "processing"
                and exec_defaults["resources_mode"] == "max-throughput"
            ):
                _wq_queue.specify_category_mode(
                    category, wq.WORK_QUEUE_ALLOCATION_MODE_MAX_THROUGHPUT
                )

        # enable fast termination of workers
        if (
            exec_defaults["fast_terminate_workers"]
            and exec_defaults["fast_terminate_workers"] > 1
        ):
            _wq_queue.activate_fast_abort_category(
                category, exec_defaults["fast_terminate_workers"]
            )


def _submit_proc_task(
    fn_wrapper,
    infile_function,
    items,
    chunksize,
    update_chunksize,
    tmpdir,
    exec_defaults,
):
    if update_chunksize:
        item = items.send(chunksize)
        exec_defaults["updated_chunksize"] = chunksize
    else:
        item = next(items)

    task = ProcCoffeaWQTask(fn_wrapper, infile_function, item, tmpdir, exec_defaults)
    task_id = _wq_queue.submit(task)
    _vprint(
        "submitted processing task id {} item {}, with {} events",
        task_id,
        task.itemid,
        len(task),
    )

    return task


def _submit_accum_tasks(
    fn_wrapper,
    infile_function,
    tasks_to_accumulate,
    force_last_accum,
    tmpdir,
    exec_defaults,
):

    chunks_per_accum = exec_defaults["chunks_per_accum"]
    chunks_accum_in_mem = exec_defaults["chunks_accum_in_mem"]

    if chunks_per_accum < 2 or chunks_accum_in_mem < 2:
        raise RuntimeError("A minimum of two chunks should be used when accumulating")

    for next_to_accum in _group_lst(tasks_to_accumulate, chunks_per_accum):
        # return immediately if not enough for a single accumulation
        if len(next_to_accum) < 2:
            return next_to_accum

        if len(next_to_accum) < chunks_per_accum and not force_last_accum:
            # not enough tasks for a chunks_per_accum, and not all events have
            # been processed.
            return next_to_accum

        accum_task = AccumCoffeaWQTask(
            fn_wrapper, infile_function, next_to_accum, tmpdir, exec_defaults
        )
        task_id = _wq_queue.submit(accum_task)
        _vprint(
            "submitted accumulation task id {} item {}, with {} events",
            task_id,
            accum_task.itemid,
            len(accum_task),
        )

    # if we get here all tasks in tasks_to_accumulate were included in an
    # accumulation.
    return []


def _group_lst(lst, n):
    """Split the lst into sublists of len n."""
    return (lst[i : i + n] for i in range(0, len(lst), n))


def _create_fn_wrapper(x509_proxy=None, prefix_name="fn_wrapper", tmpdir=None):
    """Writes a wrapper script to run dilled python functions and arguments.
    The wrapper takes as arguments the name of three files: function, argument, and output.
    The files function and argument have the dilled function and argument, respectively.
    The file output is created (or overwritten), with the dilled result of the function call.
    The wrapper created is created/deleted according to the lifetime of the work_queue_executor."""

    proxy_basename = ""
    if x509_proxy:
        proxy_basename = basename(x509_proxy)

    contents = textwrap.dedent(
        """\
                    #!/usr/bin/env python3
                    import os
                    import sys
                    import dill
                    import coffea

                    if "{proxy}":
                        os.environ['X509_USER_PROXY'] = "{proxy}"

                    (fn, args, out) = sys.argv[1], sys.argv[2], sys.argv[3]

                    with open(fn, 'rb') as f:
                        exec_function = dill.load(f)
                    with open(args, 'rb') as f:
                        exec_args = dill.load(f)

                    pickle_out = exec_function(*exec_args)
                    with open(out, 'wb') as f:
                        dill.dump(pickle_out, f)

                    # Force an OS exit here to avoid a bug in xrootd finalization
                    os._exit(0)
                    """
    )
    with tempfile.NamedTemporaryFile(prefix=prefix_name, dir=tmpdir, delete=False) as f:
        f.write(contents.format(proxy=proxy_basename).encode())
        return f.name


def _function_to_file(function, prefix_name=None, tmpdir=None):
    with tempfile.NamedTemporaryFile(
        prefix=prefix_name, suffix="_fn.p", dir=tmpdir, delete=False
    ) as f:
        dill.dump(function, f)
        return f.name


def _get_x509_proxy(x509_proxy=None):
    if x509_proxy:
        return x509_proxy

    x509_proxy = os.environ.get("X509_USER_PROXY", None)
    if x509_proxy:
        return x509_proxy

    x509_proxy = join(
        os.environ.get("TMPDIR", "/tmp"), "x509up_u{}".format(os.getuid())
    )
    if os.path.exists(x509_proxy):
        return x509_proxy

    return None


def _make_progress_bars(exec_defaults):
    items_total = exec_defaults["events_total"]
    status = exec_defaults["status"]
    unit = exec_defaults["unit"]
    bar_format = exec_defaults["bar_format"]
    chunksize = exec_defaults["updated_chunksize"]
    chunks_per_accum = exec_defaults["chunks_per_accum"]

    submit_bar = tqdm(
        total=items_total,
        disable=not status,
        unit=unit,
        desc="Submitted",
        bar_format=bar_format,
        miniters=1,
    )

    processed_bar = tqdm(
        total=items_total,
        disable=not status,
        unit=unit,
        desc="Processing",
        bar_format=bar_format,
    )

    accumulated_bar = tqdm(
        total=1 + int(items_total / (chunksize * chunks_per_accum)),
        disable=not status,
        unit="task",
        desc="Accumulated",
        bar_format=bar_format,
    )

    return {
        "submit": submit_bar,
        "process": processed_bar,
        "accumulate": accumulated_bar,
    }


def _check_dynamic_chunksize_targets(targets):
    if targets:
        for k in targets:
            if k not in ["wall_time", "memory"]:
                raise KeyError("dynamic chunksize resource {} is unknown.".format(k))


class ResultUnavailable(Exception):
    pass


class VerbosePrint:
    def __init__(self, status_mode=True, verbose_mode=True):
        self.status_mode = status_mode
        self.verbose_mode = verbose_mode

    def __call__(self, format_str, *args, **kwargs):
        if self.verbose_mode:
            self.printf(format_str, *args, **kwargs)

    def print(self, msg):
        if self.status_mode:
            tqdm.write(msg)
        else:
            print(msg)

    def printf(self, format_str, *args, **kwargs):
        msg = format_str.format(*args, **kwargs)
        self.print(msg)


_vprint = VerbosePrint()


def _floor_to_pow2(value):
    if value < 1:
        return 1
    return pow(2, math.floor(math.log2(value)))


def _compute_chunksize(task_reports, exec_defaults, sample=True):
    targets = exec_defaults["dynamic_chunksize"]

    chunksize_default = exec_defaults["chunksize"]
    chunksize_time = None
    chunksize_memory = None

    if targets is not None and len(task_reports) > 1:
        target_time = targets.get("wall_time", None)
        if target_time:
            chunksize_time = _compute_chunksize_target(
                target_time, [(time, evs) for (evs, time, mem) in task_reports]
            )

        target_memory = targets["memory"]
        if target_memory:
            chunksize_memory = _compute_chunksize_target(
                target_memory, [(mem, evs) for (evs, time, mem) in task_reports]
            )

    candidate_sizes = [c for c in [chunksize_time, chunksize_memory] if c]
    if candidate_sizes:
        chunksize = min(candidate_sizes)
    else:
        chunksize = chunksize_default

    try:
        chunksize = int(_floor_to_pow2(chunksize))
        if sample:
            # sample between value found and one minue, to better explore the
            # space.  we take advantage of the fact that the function that
            # generates chunks tries to have equally sized work units per file.
            # Most files have a different number of events, which is unlikely
            # to be a multiple of the chunsize computed. Just in case all files
            # have a multiple of the chunsize, we return chunksize - 1 half the
            # time.
            chunksize = random.choice([chunksize, max(chunksize - 1, 1)])
    except ValueError:
        chunksize = chunksize_default

    return chunksize


def _compute_chunksize_target(target, pairs):
    # if no info to compute dynamic chunksize (e.g. they info is -1), return nothing
    if len(pairs) < 1 or pairs[0][0] < 0:
        return None

    avgs = [e / max(1, target) for (target, e) in pairs]
    quantiles = numpy.quantile(avgs, [0.25, 0.5, 0.75], interpolation="nearest")

    # remove outliers below the 25%
    pairs_filtered = []
    for (i, avg) in enumerate(avgs):
        if avg >= quantiles[0]:
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
        # we assume that chunksize and target have a positive
        # correlation, with a non-negative overhead (-intercept/slope). If
        # this is not true because noisy data, use the avg chunksize/time.
        # slope and intercept may be nan when data falls in a vertical line
        # (specially at the start)
        slope = quantiles[1]
        intercept = 0

    org = (slope * target) + intercept

    return org
