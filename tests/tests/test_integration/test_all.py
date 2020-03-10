import pandas as pd

from fshank import (
    parse_fstab,
    from_dataframe,
)

def test_parse_file(shared_datadir):
    fstab_str = (shared_datadir / 'test0.fstab').read_text()

    fstab = parse_fstab(fstab_str)

def test_df(shared_datadir):

    fstab_str = (shared_datadir / 'test0.fstab').read_text()
    fstab = parse_fstab(fstab_str)

    fstab.to_df()

def test_render(shared_datadir):
    fstab_str = (shared_datadir / 'test0.fstab').read_text()
    fstab = parse_fstab(fstab_str)

    fstab.render()

def test_from_dataframe(shared_datadir):
    df_path = (shared_datadir / 'test0.csv')

    fstab_df = pd.read_csv(df_path)

    fstab = from_dataframe(fstab_df)
