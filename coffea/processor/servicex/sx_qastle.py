import ast
from typing import cast

from func_adl import EventDataset
from func_adl.object_stream import ObjectStream
from qastle import python_ast_to_text_ast
from servicex import ServiceXDataset

# We are asking func_adl_servicex to do something that I'd not thought of before - and the API I've got now
# is not actually the best looking for it (e.g. not composable). So this code below is cut-pasted from libraries, and
# at some point needs to be put back into libraries and called from there, rather than from here.
# A todo/issue when we get this working.
# As a result, it also have more limitations.


class FuncAdlDataset(EventDataset):
    "func_adl data source that will return `qastle` string for the query, providing low level access."

    def __init__(self):
        super().__init__()

    async def execute_result_async(self, a: ast.AST):
        """We will use generate the query.
        WARNING: this code is fragile - the ast above must end with an invocation of AsROOTTTree!!
        WARNING: Really will only work for xAOD backend due to separate logic required for each backend.

        This code was stolen from the `ServiceX.py` file located in `func_adl_servicex`
        """
        source = a
        if cast(ast.Name, a.func).id != "ResultTTree":
            raise Exception("Must be a call to AsROOTTtree at end of query for now")

        # Get the qastle we are going to use!
        return python_ast_to_text_ast(source)


def sx_event_stream(did: str, query: ObjectStream):
    """Fetch the data from the queue and return an async stream
    of the results (minio urls)

    Args:
        did ([type]): Dataset identifier
        query ([type]): The actual query, as an ObjectStream

    Returns:
        An async generator that can be processed with the `aiostream` library
        or an `async for` python pattern.
    """
    sx = ServiceXDataset(did, backend_type="xaod")
    return sx.get_data_rootfiles_minio_async(query.value())
