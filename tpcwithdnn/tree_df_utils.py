"""
Convert between ROOT files and Python pandas dataframes
"""
# pylint: disable=consider-using-f-string

import re
import pandas as pd
import uproot3
import uproot

from RootInteractive.Tools.aliTreePlayer import LoadTrees, tree2Panda


def pandas_to_tree(data, file_name, tree_name):
    """
    Save pandas dataframe as a ROOT TTree.

    :param pandas.DataFrame data: dataframe to be stored
    :param str file_name: path and name of the output file
    :param str tree_name: name of the result TTree
    """
    branch_dict = {data.columns[i]: data.dtypes[i]
                   for i in range(0, len(data.columns))}
    with uproot3.recreate(file_name) as file_output:
        file_output[tree_name] = uproot3.newtree(branches=branch_dict, title=tree_name)
        file_output[tree_name].extend({data.columns[i]:
                                       data[data.columns[i]].to_numpy()
                                       for i in range(0, len(data.columns))})


def tree_to_pandas_ri(file_name, tree_name, columns, exclude_columns=[], **kwargs):  # pylint: disable=dangerous-default-value
    """
    Read a TTree from a ROOT file and convert it to a pandas dataframe using RootInteractive.

    :param str file_name: path to the root file
    :param str tree_name: name of the TTree
    :param list[str] columns: names of branches or aliases to be read.
                              They can also be regular expression like ['.*'] for all branches
                              or ['.*fluc'] for all branches containing the string "fluc".
    :param list[str] exclude_columns: optional names of branches or aliases to be ignored.
                                      They can also be regular expression like ['.*fluc']
                                      for all branches containing the string "fluc".
    :return: dataframe with the specified columns
    :rtype: pandas.DataFrame
    """
    # The tree2panda function only works if the returned df has less than 1M rows.
    # For entries at rows > 1M the data is corrupt. RootInteractive developers are aware.
    max_rows_pandas = 1000000

    # LoadTrees()
    # Param 1: shell command to obtain file names, e.g. ls file.root or cat files.list
    # Param 2: input trees to be selected
    # Param 3: input tree not to be selected
    # Param 4: if not empty, files from first argument to be used
    # Param 5: verbosity level
    tree, _, _ = LoadTrees("ls %s" % file_name, tree_name, "xxx", "", 0)

    data = pd.DataFrame()
    n_entries = tree.GetEntries()
    first_entry = 0
    exclude_columns.append("%s*" % tree_name)
    while first_entry < n_entries:
        df_tmp = tree2Panda(tree, columns, "", exclude=exclude_columns,
                            nEntries=max_rows_pandas - 1, firstEntry=first_entry, **kwargs)
        data = pd.concat([data, df_tmp], ignore_index=True)
        first_entry = first_entry + max_rows_pandas - 1

    return data


def tree_to_pandas(file_name, tree_name, **kwargs):
    """
    Read a TTree from a ROOT file and convert it to a pandas DataFrame using uproot.
    Can be used for
        - TTrees with branches of single values.
        - TTrees containing std::vectors. Vectors of selected branches have to be of the same size.
        - TTrees with a combination of std::vector and single value branches.

    :param str file_name: path to the root file
    :param str tree_name: name of the TTree
    :param dictionary kwargs: optional parameters to filter branches
        - list[str] 'columns': names of branches to be read from the tree.
        - str 'filter_name': regex string in "/pattern/i" syntax to select branches to be read from
                             the tree.
        - str 'exclude': names of branches or aliases to be ignored.
                         They can also be regular expression like '.*fluc'
                         for all branches containing the string "fluc".
    :return: dataframe with the specified columns
    :rtype: pandas.DataFrame
    """
    options = {'columns': None,
               'filter_name': None,
               'exclude': None}
    options.update(kwargs)
    data = uproot.open("%s:%s" % (file_name, tree_name)).arrays(options['columns'],
                                                                filter_name=options['filter_name'],
                                                                library='pd')

    # check whether a single pandas DataFrame is returned or a tuple of DataFrames most likely due
    # to vectors of different length in the TTree
    if isinstance(data, tuple):
        raise TypeError("Return value is a tuple of DataFrames. " + \
            "Please check that vectors of selected branches have the same size.")

    # drop entry and subentry columns from std::vectors and reset index
    if data.index.nlevels > 1:
        data = data.droplevel('entry').reset_index().drop(columns="subentry")

    # exclude columns
    if options['exclude']:
        data = data.filter([col for col in data.columns
                            if not re.compile(options['exclude']).match(col)])
    return data
