# Copyright (c) 2019, IRIS-HEP
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from abc import ABC, abstractmethod
from typing import Any, Callable, AsyncGenerator
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

import aiostream
import uproot
from servicex import StreamInfoUrl
from ..accumulator import async_accumulate


class Executor(ABC):
    @abstractmethod
    def run_async_analysis(
        self,
        file_url: str,
        tree_name: str,
        process_func: Callable,
    ):
        raise NotImplementedError

    def get_result_file_stream(self, datasource):
        return datasource.stream_result_file_urls()

    async def execute(self, analysis, datasource):
        """
        Launch an analysis against the given dataset on the implementation's task framework
        :param analysis:
            The analysis to run
        :param datasource:
            The datasource to run against
        :return:
            Stream of up to date histograms. Grows as each result is received
        """
        # Stream transformed file references from ServiceX
        result_file_stream = self.get_result_file_stream(datasource)

        # Launch a task against this file
        func_results = self.launch_analysis_tasks_from_stream(
            result_file_stream, analysis.process
        )

        # Wait for all the data to show up
        async def inline_wait(r):
            "This could be inline, but python 3.6"
            x = await r
            return x

        finished_events = aiostream.stream.map(func_results, inline_wait, ordered=False)
        # Finally, accumulate!
        # There is an accumulate pattern in the aiostream lib
        async with finished_events.stream() as streamer:
            async for results in async_accumulate(streamer):
                yield results

    async def launch_analysis_tasks_from_stream(
        self,
        result_file_stream: AsyncGenerator[StreamInfoUrl, None],
        process_func: Callable,
    ) -> AsyncGenerator[Any, None]:
        """
        Invoke the implementation's task runner on each file from the serviceX stream.
        We don't know the file's tree name in advance, so grab a sample the first time
        around to inspect the tree name
        :param result_file_stream:
        :param accumulator:
        :param process_func:
        :return:
        """
        tree_name = None
        async for sx_data in result_file_stream:
            file_url = sx_data.url

            # Parse the absolute path out if this is a file:// uri. THis is due to a bug
            # in uproot4 that means `file://` isn't parsed correctly on windows.
            # TODO: Remove this hack when `uproot4` has been updated.
            p = urlparse(file_url)
            if p.scheme == "file":
                file_url = url2pathname(unquote(p.path))

            # Determine the tree name if we've not gotten it already
            if tree_name is None:
                with uproot.open(file_url) as sample:
                    tree_name = sample.keys()[0]

            # Invoke the implementation's task launcher
            data_result = self.run_async_analysis(
                file_url=file_url,
                tree_name=tree_name,
                process_func=process_func,
            )

            # Pass this down to the next item in the stream.
            yield data_result


def run_coffea_processor(
    events_url: str, tree_name: str, proc, explicit_func_pickle=False
):
    """
    Process a single file from a tree via a coffea processor on the remote node
    :param events_url:
        a URL to a ROOT file that uproot4 can open
    :param tree_name:
        The tree in the ROOT file to use for our data
    :param accumulator:
        Accumulator to store the results
    :param proc:
        Analysis function to execute. Must have signature
    :param explicit_func_pickle: bool
        Do we need to use dill to explicitly pickle the process function, or can we
        rely on the remote execution framework to handle it correctly?
    :return:
        Populated accumulator
    """
    # Since we execute remotely, explicitly include everything we need.
    from coffea.nanoevents import NanoEventsFactory
    from coffea.nanoevents.schemas.schema import auto_schema

    # Use NanoEvents to build a 4-vector
    events = NanoEventsFactory.from_root(
        file=str(events_url),
        treepath=f"/{tree_name}",
        schemaclass=auto_schema,
        metadata={"dataset": "mc15x", "filename": str(events_url)},
    ).events()

    if explicit_func_pickle:
        import dill as pickle

        f = pickle.loads(proc)
        return f(events)
    else:
        return proc(events)
