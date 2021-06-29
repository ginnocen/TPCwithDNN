# pylint: disable=pointless-string-statement
import re
import pandas as pd
import uproot3
from RootInteractive.Tools.aliTreePlayer import LoadTrees, tree2Panda


def pandas_to_tree(data, file_name, tree_name):
    """
      Parameters
      ----------
      data : pandas.DataFrame
          Data frame which should be stored as TTree
      file_name : str
          Path and name of root file
      tree_name : str
          Name of TTree
    """
    branch_dict = {data.columns[i]: data.dtypes[i]
                   for i in range(0, len(data.columns))}
    with uproot3.recreate(file_name) as file_output:
        file_output[tree_name] = uproot3.newtree(branches=branch_dict, title=tree_name)
        file_output[tree_name].extend({data.columns[i]:
                                       data[data.columns[i]].to_numpy()
                                       for i in range(0, len(data.columns))})


def tree_to_pandas_ri(file_name, tree_name, columns, exclude_columns=[]):  # pylint: disable=dangerous-default-value
    """
        Parameters
        ----------
        file_name : str
            Path to root file
        tree_name : str
            Name of TTree
        columns: sequence of str
            Names of branches or aliases to be read. Can also be regular expression like ['.*']
            for all branches or ['.*fluc'] for all branches containing the string "fluc".
        exclude_columns: sequence of str, optional
            Names of branches or aliases to be ignored. Can also be regular expression like
            ['.*fluc'] for all branches containing the string "fluc".

        Returns
        --------
        pandas.DataFrame
            Data frame with specified columns
    """
    # The tree2panda function only works if the returned df has less than 1M rows.
    # For entries at rows > 1M the data is corrupt. RootInteractive developers are aware.
    max_rows_pandas = 1000000

    tree, _, _ = LoadTrees("ls %s" % file_name, tree_name, "xxx", "", 0)
    """
    LoadTrees()
    Param 1: shell command to obtain file names, e.g. ls file.root or cat files.list
    Param 2: input trees to be selected
    Param 3: input tree not to be selected
    Param 4: if not empty, files from first argument to be used
    Param 5: verbosity level
    """

    data = pd.DataFrame()
    n_entries = tree.GetEntries()
    first_entry = 0
    exclude_columns.append("%s*" % tree_name)
    while first_entry < n_entries:
        df_tmp = tree2Panda(tree, columns, "", exclude=exclude_columns,
                            nEntries=max_rows_pandas - 1, firstEntry=first_entry)
        data = pd.concat([data, df_tmp], ignore_index=True)
        first_entry = first_entry + max_rows_pandas - 1

    return data


def tree_to_pandas(file_name, tree_name, columns, exclude_columns=""):  # pylint: disable=dangerous-default-value
    """
        Parameters
        ----------
        file_name : str
            Path to root file
        tree_name : str
            Name of TTree
        columns: sequence of str
            Names of branches or aliases to be read. Can also use wildcards like ["*"]
            for all branches or ["*fluc*"] for all branches containing the string "fluc".
        exclude_columns: str
            Regular expression of columns to be ignored, e.g. ".*Id".

        Returns
        --------
        pandas.DataFrame
            Data frame with specified columns
    """
    with uproot3.open(file_name) as file:
        data = file[tree_name].pandas.df(columns)
    if exclude_columns != "":
        data = data.filter([col for col in data.columns
                            if not re.compile(exclude_columns).match(col)])
    return data
