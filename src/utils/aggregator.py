import polars as pl


class Aggregator:
    @staticmethod
    def get_exprs(df: pl.LazyFrame) -> list[pl.Series]:
        """
        Combines expressions for maximum, mean, and variance calculations.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - list[pl.Series]: List of combined expressions.
        """
        exprs = Aggregator.max_expr(df) + Aggregator.mean_expr(df) + Aggregator.var_expr(df)

        return exprs

    @staticmethod
    def max_expr(df: pl.LazyFrame) -> list[pl.Series]:
        """
        Generates expressions for calculating maximum values for specific columns.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - list[pl.Series]: List of expressions for maximum values.
        """
        cols: list[str] = [
            col for col in df.columns if (col[-1] in ("P", "M", "A", "D", "T", "L")) or ("num_group" in col)
        ]

        expr_max: list[pl.Series] = [pl.col(col).max().alias(f"max_{col}") for col in cols]

        return expr_max

    @staticmethod
    def min_expr(df: pl.LazyFrame) -> list[pl.Series]:
        """
        Generates expressions for calculating minimum values for specific columns.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - list[pl.Series]: List of expressions for minimum values.
        """
        cols: list[str] = [
            col for col in df.columns if (col[-1] in ("P", "M", "A", "D", "T", "L")) or ("num_group" in col)
        ]

        expr_min: list[pl.Series] = [pl.col(col).min().alias(f"min_{col}") for col in cols]

        return expr_min

    @staticmethod
    def mean_expr(df: pl.LazyFrame) -> list[pl.Series]:
        """
        Generates expressions for calculating mean values for specific columns.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - list[pl.Series]: List of expressions for mean values.
        """
        cols: list[str] = [col for col in df.columns if col.endswith(("P", "A", "D"))]

        expr_mean: list[pl.Series] = [pl.col(col).mean().alias(f"mean_{col}") for col in cols]

        return expr_mean

    @staticmethod
    def var_expr(df: pl.LazyFrame) -> list[pl.Series]:
        """
        Generates expressions for calculating variance for specific columns.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - list[pl.Series]: List of expressions for variance.
        """
        cols: list[str] = [col for col in df.columns if col.endswith(("P", "A", "D"))]

        expr_var: list[pl.Series] = [pl.col(col).var().alias(f"var_{col}") for col in cols]

        return expr_var

    @staticmethod
    def mode_expr(df: pl.LazyFrame) -> list[pl.Series]:
        """
        Generates expressions for calculating mode values for specific columns.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - list[pl.Series]: List of expressions for mode values.
        """
        cols: list[str] = [col for col in df.columns if col.endswith("M")]

        expr_mode: list[pl.Series] = [pl.col(col).drop_nulls().mode().first().alias(f"mode_{col}") for col in cols]

        return expr_mode
