"""Read a Feather file (Arrow IPC file format) into a LoadedDataset.

Feather V2 is the Arrow IPC file format, which is why ``.arrow`` files are read here too.
The old Feather V1 is also read (pyarrow deprecates it but still reads it). Like Parquet, the
file stores typed columns and there is nothing to detect: ``FeatherOptions`` is empty and only
marks the format. Compressed files (LZ4, ZSTD) are decompressed by pyarrow. The bytes are read
without a copy (``pa.BufferReader``) and converted by ``arrow_table.to_frame``.

Arrow IPC *streams* (``.arrows``, the other Arrow format) are not files and are not read.
"""

from dataclasses import dataclass

import pyarrow as pa
import pyarrow.feather as feather

from litmetrics.core.io import arrow_table
from litmetrics.core.io.base import LoadedDataset, ReadError, Source

# Name of the format, shown to the user.
FORMAT_NAME = "Feather"

# First bytes of a Feather V2 / Arrow IPC file, and of a Feather V1 file.
_MAGICS = (b"ARROW1", b"FEA1")


@dataclass(frozen=True)
class FeatherOptions:
    """How a Feather file is read. Nothing to choose: the file says it all."""


def detect_options(data: bytes) -> FeatherOptions:
    """Return the options of any Feather file. Never raises: the loader checks the file."""
    return FeatherOptions()


class FeatherLoader:
    """``Loader`` for Feather (V1 and V2) and Arrow IPC files."""

    def load(self, source: Source, options: FeatherOptions) -> LoadedDataset:
        """Read the whole file in ``source``.

        Raises:
            ReadError: The file is empty, is not a Feather file, is damaged, or has no rows or
                no columns.
        """
        name = source.name
        if not source.data:
            raise ReadError(f"'{name}' is empty.")
        # Checked first: pyarrow's message for another format is "Not an Arrow file".
        if not source.data.startswith(_MAGICS):
            raise ReadError(f"'{name}' is not a Feather (Arrow IPC) file.")
        try:
            table = feather.read_table(pa.BufferReader(source.data))
            changed = arrow_table.changes(table.schema)
            df, details = arrow_table.to_frame(table)
        # ArrowException: damaged or truncated file. OSError: pyarrow raises it for some
        # truncated buffers.
        except (pa.ArrowException, OSError) as e:
            raise ReadError(f"'{name}' is damaged and cannot be read: {e}") from e
        finally:
            # R-L5: give back to the system what Arrow kept for the next read.
            pa.default_memory_pool().release_unused()

        if df.shape[1] == 0:
            raise ReadError(f"'{name}' has no columns.")
        if df.empty:
            raise ReadError(f"'{name}' has no data rows.")
        return LoadedDataset(df, name, FORMAT_NAME, details, changed=changed)
