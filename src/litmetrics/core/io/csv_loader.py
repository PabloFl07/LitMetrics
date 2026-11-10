"""Read a CSV file into a LoadedDataset.

The pyarrow engine is used first: it is about 3.7x faster than pandas' C reader (R-L1 in
docs/performance.md). When pyarrow cannot read a file (ragged rows, an option it does not
support...), the file is read again with the C engine, which also gives the clearer error
messages shown to the user.

Memory: the pyarrow engine allocates through Arrow's memory pool, which keeps the freed blocks
for later reads instead of giving them back. After reading a 569 MB file the process held
1.9 GB more for a 525 MB DataFrame; releasing the pool after each read gives back about 1 GB.
"""

import io
from typing import Literal

import pandas as pd
import pyarrow as pa

from litmetrics.core.io.base import LoadedDataset, ReadError, Source
from litmetrics.core.io.sniffing import SAMPLE_BYTES, CsvOptions


class CsvLoader:
    """``Loader`` for delimited text files (CSV, TSV, ...).

    The options are not guessed here: they come from ``sniffing.detect_options``, so the
    caller can show them or, in the future, let the user change them.
    """

    def load(self, source: Source, options: CsvOptions) -> LoadedDataset:
        """Parse ``source`` with ``options``.

        Raises:
            ReadError: The file is empty, binary, has broken encoding or is not a valid CSV.
                The message explains the problem to the user.
        """
        name = source.name
        if not source.data.strip():
            raise ReadError(f"'{name}' is empty.")
        # UTF-16 text has null bytes; in any other encoding they mean a binary file.
        if not options.encoding.startswith("utf-16") and b"\x00" in source.data[:SAMPLE_BYTES]:
            raise ReadError(
                f"'{name}' is not a text file. It may be an Excel file renamed to .csv."
            )

        try:
            try:
                df = _read(source.data, options, "pyarrow")
            # ValueError covers pandas' ParserError and the options pyarrow does not support.
            except ValueError, pa.ArrowException:
                df = _read(source.data, options, "c")
        # The encoding was checked on the whole file, so this only happens if parts of the
        # file use different encodings.
        except UnicodeDecodeError as e:
            raise ReadError(
                f"'{name}' mixes text encodings. Save it as UTF-8 and upload it again."
            ) from e
        except pd.errors.EmptyDataError as e:
            raise ReadError(f"'{name}' is empty.") from e
        except pd.errors.ParserError as e:
            raise ReadError(f"'{name}' is malformed: {e}") from e
        finally:
            # Also after a failed read. Both engines allocate through Arrow: pandas 3 stores
            # text columns as Arrow strings.
            pa.default_memory_pool().release_unused()

        # Only a header row, or only blank lines.
        if df.empty:
            raise ReadError(f"'{name}' has no data rows.")
        if not options.has_header:
            df.columns = [f"column_{i}" for i in range(1, df.shape[1] + 1)]
        return LoadedDataset(
            df,
            name,
            options.separator,
            options.encoding,
            has_header=options.has_header,
            decimal=options.decimal,
        )


def _read(data: bytes, options: CsvOptions, engine: Literal["pyarrow", "c"]) -> pd.DataFrame:
    """Parse ``data`` with one of pandas' engines. Errors are left to the caller."""
    return pd.read_csv(
        io.BytesIO(data),
        engine=engine,
        sep=options.separator,
        encoding=options.encoding,
        # header=None: pandas names the columns 0, 1, 2...; the caller renames them.
        header=0 if options.has_header else None,
        decimal=options.decimal,
        # C engine only (pyarrow rejects it): read the whole file before choosing each dtype.
        # Otherwise pandas reads it in chunks and a column can end up with mixed types.
        **({"low_memory": False} if engine == "c" else {}),
    )
