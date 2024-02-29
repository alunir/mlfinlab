import pandas as pd


def heikin_ashi(
    df,
    open: str = "Open",
    high: str = "High",
    low: str = "Low",
    close: str = "Close",
):
    heikin_ashi_df = pd.DataFrame(
        index=df.index,
        columns=[f"ha_{open}", f"ha_{high}", f"ha_{low}", f"ha_{close}"],
        dtype=float,
    )

    heikin_ashi_df[f"ha_{close}"] = (df[open] + df[high] + df[low] + df[close]) / 4

    for i in range(len(df)):
        if i == 0:
            heikin_ashi_df.iat[0, 0] = df[open].iloc[0]
        else:
            heikin_ashi_df.iat[i, 0] = (
                heikin_ashi_df.iat[i - 1, 0] + heikin_ashi_df.iat[i - 1, 3]
            ) / 2

    heikin_ashi_df[f"ha_{high}"] = (
        heikin_ashi_df.loc[:, [f"ha_{open}", f"ha_{close}"]].join(df[high]).max(axis=1)
    )
    heikin_ashi_df[f"ha_{low}"] = (
        heikin_ashi_df.loc[:, [f"ha_{open}", f"ha_{close}"]].join(df[low]).min(axis=1)
    )
    return heikin_ashi_df
