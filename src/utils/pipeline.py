import polars as pl


class Pipeline:
    @staticmethod
    def filter_cols(df: pl.DataFrame) -> pl.DataFrame:
        """
        Filters columns in the DataFrame based on null percentage and unique values for string columns.

        Args:
        - df (pl.DataFrame): Input DataFrame.

        Returns:
        - pl.DataFrame: DataFrame with filtered columns.
        """
        for col in df.columns:
            if col not in ["case_id", "year", "month", "week_num", "target"]:
                null_pct = df[col].is_null().mean()

                if null_pct > 0.95:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["case_id", "year", "month", "week_num", "target"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()

                if (freq > 200) | (freq == 1):
                    df = df.drop(col)

        return df

    @staticmethod
    def transform_cols(df: pl.DataFrame) -> pl.DataFrame:
        """
        Transforms columns in the DataFrame according to predefined rules.

        Args:
        - df (pl.DataFrame): Input DataFrame.

        Returns:
        - pl.DataFrame: DataFrame with transformed columns.
        """
        if "riskassesment_302T" in df.columns:
            if df["riskassesment_302T"].dtype == pl.Null:
                df = df.with_columns(
                    [
                        pl.Series("riskassesment_302T_rng", df["riskassesment_302T"], pl.UInt8),
                        pl.Series("riskassesment_302T_mean", df["riskassesment_302T"], pl.UInt8),
                    ]
                )
            else:
                pct_low: pl.Series = (
                    df["riskassesment_302T"].str.split(" - ").apply(lambda x: x[0].replace("%", "")).cast(pl.UInt8)
                )
                pct_high: pl.Series = (
                    df["riskassesment_302T"].str.split(" - ").apply(lambda x: x[1].replace("%", "")).cast(pl.UInt8)
                )

                diff: pl.Series = pct_high - pct_low
                avg: pl.Series = ((pct_low + pct_high) / 2).cast(pl.Float32)

                del pct_high, pct_low
                gc.collect()

                df = df.with_columns(
                    [
                        diff.alias("riskassesment_302T_rng"),
                        avg.alias("riskassesment_302T_mean"),
                    ]
                )

            df.drop("riskassesment_302T")

        return df

    @staticmethod
    def handle_dates(df: pl.DataFrame) -> pl.DataFrame:
        """
        Handles date columns in the DataFrame.

        Args:
        - df (pl.DataFrame): Input DataFrame.

        Returns:
        - pl.DataFrame: DataFrame with transformed date columns.
        """
        for col in df.columns:
            if col.endswith("D"):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days().cast(pl.Int32))

        df = df.rename({"MONTH": "month", "WEEK_NUM": "week_num"})

        df = df.with_columns(
            [
                pl.col("date_decision").dt.year().alias("year").cast(pl.Int16),
                pl.col("date_decision").dt.day().alias("day").cast(pl.UInt8),
            ]
        )

        return df.drop("date_decision")
