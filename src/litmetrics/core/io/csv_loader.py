"""Read a CSV file into a LoadedDataset."""

import io

import pandas as pd

from litmetrics.core.io.base import LoadedDataset, ReadError, Source
from litmetrics.core.io.sniffing import SAMPLE_BYTES, CsvOptions


class CsvLoader:
    """``Loader`` for delimited text files (CSV, TSV, ...).

    The options are not guessed here: they come from ``sniffing.``, so the
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
            df = pd.read_csv(
                io.BytesIO(source.data),
                sep=options.separator,
                encoding=options.encoding,
                # header=None: pandas names the columns 0, 1, 2...; renamed below.
                header=0 if options.has_header else None,
                # Read the whole file before choosing each dtype. Otherwise pandas reads it in
                # chunks and a column can end up with mixed types.
                low_memory=False,
            )
        # The encoding was checked on the whole file, so this only happ4ens if parts of the
        # file use different encodings.
        except UnicodeDecodeError as e:
            raise ReadError(
                f"'{name}' mixes text encodings. Save it as UTF-8 and upload it again."
            ) from e
        except pd.errors.EmptyDataError as e:
            raise ReadError(f"'{name}' is empty.") from e
        except pd.errors.ParserError as e:
            raise ReadError(f"'{name}' is malformed: {e}") from e

        # Only a header row, or only blank lines.
        if df.empty:
            raise ReadError(f"'{name}' has no data rows.")
        if not options.has_header:
            df.columns = [f"column_{i}" for i in range(1, df.shape[1] + 1)]
        return LoadedDataset(df, name, options.separator, options.encoding)
