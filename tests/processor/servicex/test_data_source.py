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
import pytest
from coffea.processor.servicex import DataSource, FuncAdlDataset


class TestDataSource:
    @pytest.mark.asyncio
    async def test_stream_result_file_urls_root(self, mocker):
        query = mocker.MagicMock(FuncAdlDataset)
        query.value = mocker.Mock(return_value="select * from events")

        dataset = MockDatset(
            urls=["http://foo.bar.com/yyy.ROOT", "http://baz.bar.com/xxx.ROOT"],
            datatype='root'
        )
        data_source = DataSource(query=query, metadata={}, datasets=[dataset])

        url_stream = [url async for url in data_source.stream_result_file_urls()]
        assert url_stream == [
            "http://foo.bar.com/yyy.ROOT",
            "http://baz.bar.com/xxx.ROOT",
        ]
        assert dataset.num_calls == 1
        assert dataset.called_query == "select * from events"

    @pytest.mark.asyncio
    async def test_stream_result_file_urls_parquet(self, mocker):
        query = mocker.MagicMock(FuncAdlDataset)
        query.value = mocker.Mock(return_value="select * from events")

        dataset = MockDatset(
            urls=["http://foo.bar.com/yyy.ROOT", "http://baz.bar.com/xxx.ROOT"],
            datatype='parquet'
        )
        data_source = DataSource(query=query, metadata={}, datasets=[dataset])

        url_stream = [url async for url in data_source.stream_result_file_urls()]
        assert url_stream == [
            "http://foo.bar.com/yyy.ROOT",
            "http://baz.bar.com/xxx.ROOT",
        ]
        assert dataset.num_calls_parquet == 1
        assert dataset.called_query == "select * from events"

    @pytest.mark.asyncio
    async def test_stream_result_files_root(self, mocker):
        query = mocker.MagicMock(FuncAdlDataset)
        query.value = mocker.Mock(return_value="select * from events")

        dataset = MockDatset(
            files=["http://foo.bar.com/yyy.ROOT", "http://baz.bar.com/xxx.ROOT"],
            datatype='root'
        )
        data_source = DataSource(query=query, metadata={}, datasets=[dataset])

        file_stream = [f async for f in data_source.stream_result_files()]
        assert file_stream == [
            "http://foo.bar.com/yyy.ROOT",
            "http://baz.bar.com/xxx.ROOT",
        ]
        assert dataset.num_calls == 1
        assert dataset.called_query == "select * from events"

    @pytest.mark.asyncio
    async def test_stream_result_files_parquet(self, mocker):
        query = mocker.MagicMock(FuncAdlDataset)
        query.value = mocker.Mock(return_value="select * from events")

        dataset = MockDatset(
            files=["http://foo.bar.com/yyy.ROOT", "http://baz.bar.com/xxx.ROOT"],
            datatype='parquet'
        )
        data_source = DataSource(query=query, metadata={}, datasets=[dataset])

        file_stream = [f async for f in data_source.stream_result_files()]
        assert file_stream == [
            "http://foo.bar.com/yyy.ROOT",
            "http://baz.bar.com/xxx.ROOT",
        ]
        assert dataset.num_calls_parquet == 1
        assert dataset.called_query == "select * from events"


class MockDatset:
    def __init__(self, files=[], urls=[], datatype='root'):
        self.files = files
        self.urls = urls
        self.called_query = None
        self.num_calls = 0
        self.num_calls_parquet = 0
        self.datatype = datatype

    async def get_data_rootfiles_url_stream(self, query):
        self.called_query = query
        self.num_calls += 1

        for url in self.urls:
            yield url

    async def get_data_parquet_url_stream(self, query):
        self.called_query = query
        self.num_calls_parquet += 1

        for url in self.urls:
            yield url

    async def get_data_rootfiles_stream(self, query):
        self.called_query = query
        self.num_calls += 1

        for f in self.files:
            yield f

    async def get_data_parquet_stream(self, query):
        self.called_query = query
        self.num_calls_parquet += 1

        for f in self.files:
            yield f

    def first_supported_datatype(self, possible_list):
        return self.datatype
