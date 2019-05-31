import time
import signal
from functools import wraps


def timeout(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        import signal
        import datetime

        def handler(signum, frame):
            raise Exception("Timeout hit")

        signal.signal(signal.SIGALRM, handler)
        if kwargs.get('timeout', None):
            signal.alarm(kwargs.get('timeout'))
            print("[Timeout wrapper:{}] Timeout {}".format(datetime.datetime.now(),
                                                           kwargs.get('timeout')))
        try:
            retval = func(*args, **kwargs)
            return retval
        except Exception as e:
            print("[Timeout wrapper:{}] Caught an exception {}".format(datetime.datetime.now(),
                                                                       e))
            raise

    return wrapper
