# Copyright (c) 2021, IRIS-HEP
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


from typing import AsyncGenerator, Dict, List

from servicex import ServiceXDataset, StreamInfoPath, StreamInfoUrl
from func_adl import ObjectStream


class DataSource:
    def __init__(
        self,
        query: ObjectStream,
        metadata: Dict[str, str] = {},
        datasets: List[ServiceXDataset] = [],
    ):
        self.query = query
        self.metadata = metadata
        self.schema = None
        self.datasets = datasets

    async def stream_result_file_urls(self) -> AsyncGenerator[StreamInfoUrl, None]:
        """Launch all datasources at once

        TODO: This is currently sync (that outter for loop does one datasource and then the next).
        Need to move to a different paradigm. Perhaps using the `aiostream` library.

        Yields:
            [type]: [description]
        """
        for dataset in self.datasets:
            data_type = dataset.first_supported_datatype(["parquet", "root"])
            if data_type == "root":
                async for file in dataset.get_data_rootfiles_url_stream(qastle):
                    yield file
            elif data_type == "parquet":
                async for file in dataset.get_data_parquet_url_stream(qastle):
                    yield file
            else:
                raise Exception(
                    f"This dataset ({str(dataset)}) supports unknown datatypes"
                )

    async def stream_result_files(self) -> AsyncGenerator[StreamInfoPath, None]:
        """Launch all datasources at once

        TODO: This is currently sync (that outter for loop does one datasource and then the next).
        Need to move to a different paradigm. Perhaps using the `aiostream` library.

        Yields:
            [type]: [description]
        """
        qastle = await self.query.value_async()
        for dataset in self.datasets:
            data_type = dataset.first_supported_datatype(["parquet", "root"])
            if data_type == "root":
                async for file in dataset.get_data_rootfiles_stream(qastle):
                    yield file
            else:
                async for file in dataset.get_data_parquet_stream(qastle):
                    yield file
