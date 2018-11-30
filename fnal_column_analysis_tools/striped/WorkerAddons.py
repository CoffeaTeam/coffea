import time

class Timer(object):
    def __init__(self):
        self.__tic = {}
        self.__toc = {}

    def set_tic(self,name):
        self.__tic[name] = time.time()

    def set_toc(self,name):
        self.__toc[name] = time.time()

    def get_tic(self,name):
        return self.__tic[name]

    def get_toc(self,name):
        return self.__toc[name]

    def tics(self):
        return self.__tic

    def tocs(self):
        return self.__toc
    
    def fill_job_timer_info(self,stripelen,job):
        tic = self.__tic
        toc = self.__toc
        for key in tic.keys():
            dtime = toc[key]-tic[key]
            job.fill(
                category=key,
                stripeThroughput=stripelen/(dtime+1e-8),
                stripeTime=dtime
            )
