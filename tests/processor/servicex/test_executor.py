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
import os
from pathlib import Path
from typing import Callable, Dict

import pytest
from servicex import StreamInfoUrl
from servicex.servicex import StreamInfoPath

from coffea.processor.servicex import Analysis
from coffea.processor.servicex.executor import Executor


class TestableExecutor(Executor):
    def __init__(self):
        self.tree_name = None

    async def run_async_analysis(
        self,
        file_url: str,
        tree_name: str,
        data_type: str,
        meta_data: Dict[str, str],
        process_func: Callable,
    ):

        # Record the tree name so we can verify it later
        self.tree_name = tree_name
        self.data_type = data_type
        self.meta_data = meta_data
        return {file_url: 1}


class MockDataSource:
    def __init__(self, urls=[]):
        self.urls = urls
        self.metadata = {'item': 'value'}

    async def stream_result_file_urls(self):
        for url in self.urls:
            yield url


async def mock_async_generator(list):
    for x in list:
        yield x


class TestExecutor:
    @pytest.mark.asyncio
    async def test_execute_root(self, mocker):
        executor = TestableExecutor()
        analysis = mocker.MagicMock(Analysis)
        mock_root_context = mocker.Mock()
        mock_uproot_open = mocker.patch(
            "coffea.processor.servicex.executor.uproot.open",
            return_value=mock_root_context,
        )

        mock_uproot_file = mocker.Mock()
        mock_uproot_file.keys = mocker.Mock(return_value=["myTree", "yourTree"])
        mock_root_context.__enter__ = mocker.Mock(return_value=mock_uproot_file)
        mock_root_context.__exit__ = mocker.Mock()

        datasource = MockDataSource(
            urls=[
                ('root', "dataset1", StreamInfoUrl("foo", "http://foo.bar/foo", "bucket")),
                ('root', "dataset1", StreamInfoUrl("foo", "http://foo.bar/foo1", "bucket")),
            ]
        )

        hist_stream = [f async for f in executor.execute(analysis, datasource)]
        assert len(hist_stream) == 2
        mock_uproot_open.assert_called_with("http://foo.bar/foo")
        assert executor.tree_name == "myTree"
        assert executor.data_type == 'root'
        assert executor.meta_data == {'dataset': 'dataset1', 'item': 'value'}

    @pytest.mark.asyncio
    async def test_execute_parquet(self, mocker):
        executor = TestableExecutor()
        analysis = mocker.MagicMock(Analysis)

        datasource = MockDataSource(
            urls=[
                ('parquet', "dataset1", StreamInfoUrl("foo", "http://foo.bar/foo", "bucket")),
                ('parquet', "dataset1", StreamInfoUrl("foo", "http://foo.bar/foo1", "bucket")),
            ]
        )

        hist_stream = [f async for f in executor.execute(analysis, datasource)]
        assert len(hist_stream) == 2
        assert executor.tree_name == None
        assert executor.data_type == 'parquet'
        assert executor.meta_data == {'dataset': 'dataset1', 'item': 'value'}

    @pytest.mark.asyncio
    async def test_execute_with_file_url_root(self, mocker):
        executor = TestableExecutor()
        analysis = mocker.MagicMock(Analysis)
        analysis.accumulator = mocker.Mock()
        mock_histogram = mocker.Mock()
        analysis.accumulator.identity = mocker.Mock(return_value=mock_histogram)

        mock_root_context = mocker.Mock()
        mock_uproot_open = mocker.patch(
            "coffea.processor.servicex.executor.uproot.open",
            return_value=mock_root_context,
        )
        mock_uproot_file = mocker.Mock()
        mock_uproot_file.keys = mocker.Mock(return_value=["myTree", "yourTree"])
        mock_root_context.__enter__ = mocker.Mock(return_value=mock_uproot_file)
        mock_root_context.__exit__ = mocker.Mock()

        file_path = (
            os.path.join(os.sep, "foo")
            if os.name != "nt"
            else os.path.join("c:", os.sep, "foo")
        )

        datsource = MockDataSource(
            urls=[
                ('root', "dataset1", StreamInfoPath(
                    "root1.ROOT", Path(os.path.join(file_path, "root1.ROOT"))
                )),
                ('root', "dataset1", StreamInfoPath(
                    "root2.ROOT", Path(os.path.join(file_path, "root2.ROOT"))
                )),
            ]
        )

        hist_stream = [f async for f in executor.execute(analysis, datsource)]

        if os.name != "nt":
            mock_uproot_open.assert_called_with("file:///foo/root1.ROOT")
        else:
            mock_uproot_open.assert_called_with("file:///c:/foo/root1.ROOT")

        assert len(hist_stream) == 2

    @pytest.mark.asyncio
    async def test_execute_with_file_url_parquet(self, mocker):
        executor = TestableExecutor()
        analysis = mocker.MagicMock(Analysis)
        analysis.accumulator = mocker.Mock()
        mock_histogram = mocker.Mock()
        analysis.accumulator.identity = mocker.Mock(return_value=mock_histogram)

        file_path = (
            os.path.join(os.sep, "foo")
            if os.name != "nt"
            else os.path.join("c:", os.sep, "foo")
        )

        datsource = MockDataSource(
            urls=[
                ('parquet', "dataset1", StreamInfoPath(
                    "root1.parquet", Path(os.path.join(file_path, "root1.parquet"))
                )),
                ('parquet', "dataset1", StreamInfoPath(
                    "root2.parquet", Path(os.path.join(file_path, "root2.parquet"))
                )),
            ]
        )

        hist_stream = [f async for f in executor.execute(analysis, datsource)]

        assert len(hist_stream) == 2
