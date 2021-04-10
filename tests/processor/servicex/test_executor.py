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
import asyncio
from pathlib import Path
from typing import Callable

import pytest
from servicex import StreamInfoUrl
from servicex.servicex import StreamInfoPath

from coffea.processor.servicex import Accumulator, Analysis
from coffea.processor.servicex.executor import Executor


class TestableExecutor(Executor):
    def __init__(self):
        self.tree_name = None

    async def run_async_analysis(self, file_url: str, tree_name: str,
                                 accumulator: Accumulator, process_func: Callable):
        # Create an async task that will tell us the file we were processing for
        # this analysis
        async def foo(payload):
            return payload

        # Record the tree name so we can verify it later
        self.tree_name = tree_name

        return asyncio.create_task(foo(file_url))


class MockDataSource:
    def __init__(self, urls=[]):
        self.urls = urls

    async def stream_result_file_urls(self):
        for url in self.urls:
            yield url


class TestExecutor:
    @pytest.mark.asyncio
    async def test_execute(self, mocker):
        executor = TestableExecutor()
        analysis = mocker.MagicMock(Analysis)

        analysis.accumulator = mocker.Mock()
        mock_histogram = mocker.Mock()
        analysis.accumulator.identity = mocker.Mock(return_value=mock_histogram)

        mock_root_context = mocker.Mock()
        mock_uproot_open = \
            mocker.patch('coffea.processor.servicex.executor.uproot.open',
                         return_value=mock_root_context)
        mock_uproot_file = mocker.Mock()
        mock_uproot_file.keys = mocker.Mock(return_value=['myTree', 'yourTree'])
        mock_root_context.__enter__ = mocker.Mock(return_value=mock_uproot_file)
        mock_root_context.__exit__ = mocker.Mock()

        datasource = MockDataSource(urls=[
            StreamInfoUrl("foo", "http://foo.bar/foo", "bucket"),
            StreamInfoUrl("foo", "http://foo.bar/foo1", "bucket")
        ])

        hist_stream = [f async for f in executor.execute(analysis, datasource)]

        mock_uproot_open.assert_called_with("http://foo.bar/foo")

        assert executor.tree_name == "myTree"

        # Each result from the execution stream should be a growing histogram from
        # the analysis object
        assert all([returned_hist == mock_histogram for returned_hist in hist_stream])

        # The histogram grows by executor calling add with each result returned
        histograms = [r[0][0].result() for r in mock_histogram.add.call_args_list]
        assert histograms == ['http://foo.bar/foo', 'http://foo.bar/foo1']

    @pytest.mark.asyncio
    async def test_execute_with_file_url(self, mocker):
        executor = TestableExecutor()
        analysis = mocker.MagicMock(Analysis)
        analysis.accumulator = mocker.Mock()
        mock_histogram = mocker.Mock()
        analysis.accumulator.identity = mocker.Mock(return_value=mock_histogram)

        mock_root_context = mocker.Mock()
        mock_uproot_open = \
            mocker.patch('coffea.processor.servicex.executor.uproot.open',
                         return_value=mock_root_context)
        mock_uproot_file = mocker.Mock()
        mock_uproot_file.keys = mocker.Mock(return_value=['myTree', 'yourTree'])
        mock_root_context.__enter__ = mocker.Mock(return_value=mock_uproot_file)
        mock_root_context.__exit__ = mocker.Mock()

        datsource = MockDataSource(urls=[
            StreamInfoPath("root1.ROOT", Path("/home/test/root1.ROOT")),
            StreamInfoPath("root2.ROOT", Path("/home/test2/root2.ROOT"))
        ])

        hist_stream = [f async for f in executor.execute(analysis, datsource)]

        mock_uproot_open.assert_called_with("/home/test/root1.ROOT")

        # Each result from the execution stream should be a growing histogram from
        # the analysis object
        assert all([returned_hist == mock_histogram for returned_hist in hist_stream])

        # The histogram grows by executor calling add with each result returned
        histograms = [r[0][0].result() for r in mock_histogram.add.call_args_list]
        assert histograms == ['/home/test/root1.ROOT', '/home/test2/root2.ROOT']
