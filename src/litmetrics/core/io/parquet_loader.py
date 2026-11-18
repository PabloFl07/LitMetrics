"""Read a Parquet file into a LoadedDataset.

Parquet stores typed columns and says how it was written, so there is nothing to detect:
``ParquetOptions`` is empty and only marks the format (``formats.load`` picks the loader by
the class of the options). The bytes are read by pyarrow without a copy (``pa.BufferReader``)
and converted by ``arrow_table.to_frame``, which also lists the columns it had to convert.
"""

from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq

from litmetrics.core.io import arrow_table
from litmetrics.core.io.base import LoadedDataset, ReadError, Source

# Name of the format, shown to the user.
FORMAT_NAME = "Parquet"

# Every Parquet file starts and ends with these 4 bytes.
_MAGIC = b"PAR1"


@dataclass(frozen=True)
class ParquetOptions:
    """How a Parquet file is read. Nothing to choose: the file says it all."""


def detect_options(data: bytes) -> ParquetOptions:
    """Return the options of any Parquet file. Never raises: the loader checks the file."""
    return ParquetOptions()


class ParquetLoader:
    """``Loader`` for Parquet files."""

    def load(self, source: Source, options: ParquetOptions) -> LoadedDataset:
        """Read the whole file in ``source``.

        Raises:
            ReadError: The file is empty, is not a Parquet file, is damaged, or has no rows or
                no columns.
        """
        name = source.name
        if not source.data:
            raise ReadError(f"'{name}' is empty.")
        # Checked first: pyarrow's message for another format talks about footers.
        if not (source.data.startswith(_MAGIC) and source.data.endswith(_MAGIC)):
            raise ReadError(f"'{name}' is not a Parquet file.")
        try:
            table = pq.read_table(pa.BufferReader(source.data))
            changed = arrow_table.changes(table.schema)
            df, details = arrow_table.to_frame(table)
        # ArrowException: damaged pages or footer. OSError: pyarrow raises it for some
        # truncated files.
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
