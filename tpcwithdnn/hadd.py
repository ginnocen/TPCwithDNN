"""
A pyROOT version of $ROOTSYS/tutorials/io/hadd.C for merging ROOT files.
"""

from ROOT import TList, TH1, TIter # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TDirectory, TTree, TChain # pylint: disable=import-error, no-name-in-module
from ROOT import gDirectory # pylint: disable=import-error, no-name-in-module

from tpcwithdnn.logger import get_logger

def merge_root_file(target, source_list):
    """
    Merge next file from the source list with the target file.
    Function called recursively for each element of the list.

    :param TFile target: the result ROOT file
    :param TList source_list: list of input files to merge
    """
    logger = get_logger()
    raw_path = target.GetPath()
    path = raw_path[raw_path.find(":")+1:]
    logger.info("Target path: %s processed: %s" % (raw_path, path))

    first_source = source_list.First()
    first_source.cd(path)
    current_source_dir = gDirectory
    # gain time, do not add the objects in the list in memory
    status = TH1.AddDirectoryStatus()
    TH1.AddDirectory(False)

    # loop over all keys in this directory
    #global_chain = TChain()
    next_key = TIter(current_source_dir.GetListOfKeys())
    #key = TKey()
    #TKey old_key = None
    key = next_key()
    while key:
        # keep only the highest cycle number for each key
        #if old_key and not old_key.GetName() == key.GetName():
        #    continue
        # read object from first source file
        first_source.cd(path)
        obj = key.ReadObj()

        if obj.IsA().InheritsFrom(TH1.Class()):
            # descendant of TH1 -> merge it
            logger.info("Merging histogram %s" % obj.GetName())
            h1 = TH1(obj)

            # loop over all source files and add the content of the
            # correspondant histogram to the one pointed to by "h1"
            next_source = source_list.After(first_source)
            while next_source:
                # make sure we are at the correct directory level by cd'ing to path
                next_source.cd(path)
                key2 = gDirectory.GetListOfKeys().FindObject(h1.GetName())
                if key2:
                    h2 = TH1(key2.ReadObj())
                    h1.Add(h2)
                    #del h2
                next_source = source_list.After(next_source)

        elif obj.IsA().InheritsFrom(TTree.Class()):
            logger.info("Merging tree %s" % obj.GetName())
            # loop over all source files and create a chain of Trees "global_chain"
            obj_name = obj.GetName()
            global_chain = TChain(obj_name)
            global_chain.Add(first_source.GetName())
            next_source = source_list.After(first_source)
            while next_source:
                global_chain.Add(next_source.GetName())
                next_source = source_list.After(next_source)

        elif obj.IsA().InheritsFrom(TDirectory.Class()):
            logger.info("Found subdirectory %s" % obj.GetName())
            # create a new subdir of same name and title in the target file
            target.cd()
            new_dir = target.mkdir(obj.GetName(), obj.GetTitle())
            # newdir is now the starting point of another round of merging
            # newdir still knows its depth within the target file via
            # GetPath(), so we can still figure out where we are in the recursion
            merge_root_file(new_dir, source_list)

        else:
            logger.info("Unknown object type, name: %s, title: %s" % (obj.GetName(), obj.GetTitle()))

        # now write the merged histogram (which is "in" obj) to the target file
        # note that this will just store obj in the current directory level,
        # which is not persistent until the complete directory itself is stored
        # by "target.Write()" below
        if obj is not None:
            target.cd()
            # if the object is a tree, it is stored in global_chain...
            if obj.IsA().InheritsFrom(TTree.Class()):
                global_chain.Merge(target.GetFile(), 0, "keep")
            else:
                obj.Write(key.GetName())

        # move to the next element
        key = next_key()

    # save modifications to target file
    target.SaveSelf(True)
    TH1.AddDirectory(status)
    target.Write()

def hadd(input_file_names, result_file_name):
    """
    The top function for merging files.

    :param str result_file_name: path to the output file
    :param list[str] source_list: list of paths to the input files
    """
    target = TFile.Open(result_file_name, "RECREATE")
    file_list = TList()
    for file_name in input_file_names:
        file_list.Add(TFile.Open(file_name))
    merge_root_file(target, file_list)
