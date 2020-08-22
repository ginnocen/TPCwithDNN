# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
# pylint: disable=fixme, too-many-statements, too-many-instance-attributes
import os
from array import array
import matplotlib
from ROOT import TFile, TTree  # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.logger import get_logger
from data_loader import load_data_original, get_event_mean_indices

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use("Agg")

class DataValidator:
    # Class Attribute
    # TODO: What is this for?
    species = "data validator"
    # FIXME: here I just copied something from the dnn_analyzer. It is likely
    # more information (e.g. the model name). You can just copy what you need
    # from the dnn_optimiser and delete what you dont need.

    def __init__(self, data_param, case):
        self.logger = get_logger()
        self.logger.info("DataValidator::Init\nCase: %s", case)

        # Dataset config
        self.grid_phi = data_param["grid_phi"]
        self.grid_z = data_param["grid_z"]
        self.grid_r = data_param["grid_r"]

        # Directories
        train_dir = data_param["dirinput_bias"] if data_param["train_bias"] \
                    else data_param["dirinput_nobias"]
        test_dir = data_param["dirinput_bias"] if data_param["test_bias"] \
                    else data_param["dirinput_nobias"]
        apply_dir = data_param["dirinput_bias"] if data_param["apply_bias"] \
                    else data_param["dirinput_nobias"]
        self.dirinput_train = "%s/SC-%d-%d-%d/" % \
                              (train_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_test = "%s/SC-%d-%d-%d/" % \
                             (test_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_apply = "%s/SC-%d-%d-%d/" % \
                              (apply_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.diroutflattree = data_param["diroutflattree"]
        self.suffix_ds = "phi%d_r%d_z%d" % \
                (self.grid_phi, self.grid_r, self.grid_z)

        # Parameters for getting input indices
        self.maxrandomfiles = data_param["maxrandomfiles"]
        self.range_mean_index = data_param["range_mean_index"]
        self.indices_events_means = None
        self.total_events = 0
        self.tree_events = data_param["tree_events"]


    def set_ranges(self, ranges, total_events):
        self.total_events = total_events

        self.indices_events_means, _ = get_event_mean_indices(
            self.maxrandomfiles, self.range_mean_index, ranges)


    # pylint: disable=too-many-locals
    def create_data(self):
        # FIXME : as you can imagine this is a complete duplication of what we
        # have in the dnn optimiser. But once this class is finished, we will
        # remove that part of code from the dnn_analyzer. Most likely also the
        # plotting code will be moved here. For the moment lets just keep the
        # code duplication.

        self.logger.info("DataValidator::create_data")
        self.logger.warning("DO YOU REALLY WANT TO DO IT? IT TAKES TIME")
        outfile_name = "%s/tree%s.root" % (self.diroutflattree, self.suffix_ds)
        myfile = TFile.Open(outfile_name, "recreate")

        tree = TTree('tvoxels', 'tree with histos')
        indexr = array('i', [0])
        indexphi = array('i', [0])
        indexz = array('i', [0])
        posr = array('f', [0])
        posphi = array('f', [0])
        posz = array('f', [0])
        evtid = array('i', [0])
        meanid = array('i', [0])
        randomid = array('i', [0])
        distmeanr = array('f', [0])
        distmeanrphi = array('f', [0])
        distmeanz = array('f', [0])
        distrndr = array('f', [0])
        distrndrphi = array('f', [0])
        distrndz = array('f', [0])
        tree.Branch('indexr', indexr, 'indexr/I')
        tree.Branch('indexphi', indexphi, 'indexphi/I')
        tree.Branch('indexz', indexz, 'indexz/I')
        tree.Branch('posr', posr, 'posr/F')
        tree.Branch('posphi', posphi, 'posphi/F')
        tree.Branch('posz', posz, 'posz/F')
        tree.Branch('distmeanr', distmeanr, 'distmeanr/F')
        tree.Branch('distmeanrphi', distmeanrphi, 'distmeanrphi/F')
        tree.Branch('distmeanz', distmeanz, 'distmeanz/F')
        tree.Branch('distrndr', distrndr, 'distrndr/F')
        tree.Branch('distrndrphi', distrndrphi, 'distrndrphi/F')
        tree.Branch('distrndz', distrndz, 'distrndz/F')
        tree.Branch('evtid', evtid, 'evtid/I')
        tree.Branch('meanid', meanid, 'meanid/I')
        tree.Branch('randomid', randomid, 'randomid/I')

        for counter, indexev in enumerate(self.indices_events_means):
            self.logger.info("processing event: %d [%d, %d]", counter, indexev[0], indexev[1])

            # TODO: Should it be for train or apply data?
            [vec_r_pos, vec_phi_pos, vec_z_pos,
             _, _,
             vec_mean_dist_r, vec_random_dist_r,
             vec_mean_dist_rphi, vec_random_dist_rphi,
             vec_mean_dist_z, vec_random_dist_z] = load_data_original(self.dirinput_apply, indexev)

            vec_r_pos = vec_r_pos.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_phi_pos = vec_phi_pos.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_z_pos = vec_z_pos.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_mean_dist_r = vec_mean_dist_r.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_random_dist_r = vec_random_dist_r.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_mean_dist_rphi = vec_mean_dist_rphi.reshape(self.grid_phi, self.grid_r,
                                                            self.grid_z*2)
            vec_random_dist_rphi = vec_random_dist_rphi.reshape(self.grid_phi, self.grid_r,
                                                                self.grid_z*2)
            vec_mean_dist_z = vec_mean_dist_z.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_random_dist_z = vec_random_dist_z.reshape(self.grid_phi, self.grid_r, self.grid_z*2)

            for cur_indexphi in range(self.grid_phi):
                for cur_indexr in range(self.grid_r):
                    for cur_indexz in range(self.grid_z*2):
                        indexphi[0] = cur_indexphi
                        indexr[0] = cur_indexr
                        indexz[0] = cur_indexz
                        posr[0] = vec_r_pos[cur_indexphi][cur_indexr][cur_indexz]
                        posphi[0] = vec_phi_pos[cur_indexphi][cur_indexr][cur_indexz]
                        posz[0] = vec_z_pos[cur_indexphi][cur_indexr][cur_indexz]
                        distmeanr[0] = vec_mean_dist_r[cur_indexphi][cur_indexr][cur_indexz]
                        distmeanrphi[0] = vec_mean_dist_rphi[cur_indexphi][cur_indexr][cur_indexz]
                        distmeanz[0] = vec_mean_dist_z[cur_indexphi][cur_indexr][cur_indexz]
                        distrndr[0] = vec_random_dist_r[cur_indexphi][cur_indexr][cur_indexz]
                        distrndrphi[0] = vec_random_dist_rphi[cur_indexphi][cur_indexr][cur_indexz]
                        distrndz[0] = vec_random_dist_z[cur_indexphi][cur_indexr][cur_indexz]
                        evtid[0] = indexev[0] + 10000*indexev[1]
                        meanid[0] = indexev[1]
                        randomid[0] = indexev[0]
                        tree.Fill()

            if counter + 1 == self.tree_events:
                break

        myfile.Write()
        myfile.Close()
        self.logger.info("Tree written in %s", outfile_name)


       # FIXME: HERE YOU WOULD NEED TO LOAD THE MODEL, APPLY THE MODEL TO THE DATA AND
       # FILL NEW COLUMNS THAT CONTAIN e.g. THE PREDICTED DISTORTION
       # FLUCTUATIONS. HERE IS WHERE THE CODE OF ERNST SHOULD BE INSERTED.
