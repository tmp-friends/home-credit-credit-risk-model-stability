from glob import glob
from pathlib import Path
import gc
import polars as pl

from .aggregator import Aggregator
from .utility import Utility


class SchemaGen:
    @staticmethod
    def change_dtypes(df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Changes the data types of columns in the DataFrame.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - pl.LazyFrame: LazyFrame with modified data types.
        """
        for col in df.columns:
            if col == "case_id":
                df = df.with_columns(pl.col(col).cast(pl.UInt32).alias(col))
            elif col in ["WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.UInt16).alias(col))
            elif col == "date_decision" or col[-1] == "D":
                df = df.with_columns(pl.col(col).cast(pl.Date).alias(col))
            elif col[-1] in ["P", "A"]:
                df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.Utf8))

        return df

    @staticmethod
    def scan_files(glob_path: str, depth: int = None) -> pl.LazyFrame:
        """
        Scans Parquet files matching the glob pattern and combines them into a LazyFrame.

        Args:
        - glob_path (str): Glob pattern to match Parquet files.
        - depth (int, optional): Depth level for data aggregation. Defaults to None.

        Returns:
        - pl.LazyFrame: Combined LazyFrame.
        """
        chunks: list[pl.LazyFrame] = []
        for path in glob(str(glob_path)):
            df: pl.LazyFrame = pl.scan_parquet(path, low_memory=True, rechunk=True).pipe(SchemaGen.change_dtypes)
            print(f"File {Path(path).stem} loaded into memory.")

            # depth>0 は集計値を使わないと、left join時にOOMになる（RAM 64GBでも)
            if depth in (1, 2):
                exprs: list[pl.Series] = Aggregator.get_exprs(df)
                df = df.group_by("case_id").agg(exprs)

                del exprs
                gc.collect()

            chunks.append(df)

        df = pl.concat(chunks, how="vertical_relaxed")

        del chunks
        gc.collect()

        df = df.unique(subset=["case_id"])

        return df

    @staticmethod
    def join_dataframes(
        df_base: pl.LazyFrame,
        depth_0: list[pl.LazyFrame],
        depth_1: list[pl.LazyFrame],
        depth_2: list[pl.LazyFrame],
    ) -> pl.DataFrame:
        """
        Joins multiple LazyFrames with a base LazyFrame.

        Args:
        - df_base (pl.LazyFrame): Base LazyFrame.
        - depth_0 (list[pl.LazyFrame]): List of LazyFrames for depth 0.
        - depth_1 (list[pl.LazyFrame]): List of LazyFrames for depth 1.
        - depth_2 (list[pl.LazyFrame]): List of LazyFrames for depth 2.

        Returns:
        - pl.DataFrame: Joined DataFrame.
        """
        for i, df in enumerate(depth_0 + depth_1 + depth_2):
            df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")

        return df_base.collect().pipe(Utility.reduce_memory_usage, "df_train")
