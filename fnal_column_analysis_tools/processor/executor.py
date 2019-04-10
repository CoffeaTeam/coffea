import concurrent.futures
import time


def iterative_executor(items, function, accumulator, status=True):
    for i, item in enumerate(items):
        accumulator += function(item)
        if status:
            print("Done processing item % 4d / % 4d" % (i + 1, len(items)))
    return accumulator


def futures_executor(items, function, accumulator, workers=2, status=True):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = set()
        try:
            futures.update(executor.submit(function, item) for item in items)
            total = len(futures)
            processed = 0
            while len(futures) > 0:
                finished = set(job for job in futures if job.done())
                for job in finished:
                    accumulator += job.result()
                    processed += 1
                    if status:
                        print("Processing: done with % 4d / % 4d items" % (processed, total))
                futures -= finished
                del finished
                time.sleep(1)
        except KeyboardInterrupt:
            for job in futures:
                job.cancel()
            if status:
                print("Received SIGINT, killed pending jobs.  Running jobs will continue to completion.")
                print("Running jobs:", sum(1 for j in futures if j.running()))
        except Exception:
            for job in futures:
                job.cancel()
            raise
    return accumulator


def condor_executor(items, function, accumulator, jobs, status=True):
    raise NotImplementedError
