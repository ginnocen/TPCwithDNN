# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=too-many-locals, too-many-arguments, fixme
import random
import numpy as np

def load_data_original(input_data, event_index):
    files = ["%sdata/Pos/0-vecRPos.npy" % input_data,
             "%sdata/Pos/0-vecPhiPos.npy" % input_data,
             "%sdata/Pos/0-vecZPos.npy" % input_data,
             "%sdata/Mean/%d-vecMeanSC.npy" % (input_data, event_index[1]),
             "%sdata/Random/%d-vecRandomSC.npy" % (input_data, event_index[0]),
             "%sdata/Mean/%d-vecMeanDistR.npy" % (input_data, event_index[1]),
             "%sdata/Random/%d-vecRandomDistR.npy" % (input_data, event_index[0]),
             "%sdata/Mean/%d-vecMeanDistRPhi.npy" % (input_data, event_index[1]),
             "%sdata/Random/%d-vecRandomDistRPhi.npy" % (input_data, event_index[0]),
             "%sdata/Mean/%d-vecMeanDistZ.npy" % (input_data, event_index[1]),
             "%sdata/Random/%d-vecRandomDistZ.npy" % (input_data, event_index[0])]

    return [np.load(f) for f in files]


def load_data(input_data, event_index, selopt_input, selopt_output):

    """ Here we define the functionalties to load the files from the input
    directory which is set in the database. Here below the description of
    the input files:
        - 0-vecZPos.npy, 0-vecRPos.npy, 0-vecPhiPos.npy contains the
        position of the FIXME. There is only one of these files for each
        folder, therefore for each bunch of events
        Input features for training:
        - vecMeanSC.npy: average space charge in each bin of r, rphi and z.
        - vecRandomSC.npy: fluctuation of the space charge.
        Output from the numberical calculations:
        - vecMeanDistR.npy average distorsion along the R axis in the same
          grid. It represents the expected distorsion that an electron
          passing by that region would have as a consequence of the IBF.
        - vecRandomDistR.npy are the correponding fluctuations.
        - All the distorsions along the other directions have a consistent
          naming choice.

    """

    [_, _, vec_z_pos,
     vec_mean_sc, vec_random_sc,
     vec_mean_dist_r, vec_random_dist_r,
     vec_mean_dist_rphi, vec_random_dist_rphi,
     vec_mean_dist_z, vec_random_dist_z] = load_data_original(input_data, event_index)

    # Here below we define the preselections on the input data for the training.
    # Three options are currently implemented.
    # selopt_input == 0 selects only clusters with positive z position
    # selopt_input == 1 selects only clusters with negative z position
    # selopt_input == 2 uses all data with no selections

    if selopt_input == 0:
        vec_mean_sc = vec_mean_sc[vec_z_pos >= 0]
        vec_fluctuation_sc = vec_mean_sc - vec_random_sc[vec_z_pos >= 0]
    elif selopt_input == 1:
        vec_mean_sc = vec_mean_sc[vec_z_pos < 0]
        vec_fluctuation_sc = vec_mean_sc - vec_random_sc[vec_z_pos < 0]
    elif selopt_input == 2:
        vec_fluctuation_sc = vec_mean_sc - vec_random_sc

    # selopt_output == 0 selects only clusters with positive z position
    # selopt_output == 1 selects only clusters with negative z position
    # selopt_output == 2 uses all data with no selections

    if selopt_output == 0:
        vec_fluctuation_dist_r = \
                vec_mean_dist_r[vec_z_pos >= 0] - vec_random_dist_r[vec_z_pos >= 0]
        vec_fluctuation_dist_rphi = \
                vec_mean_dist_rphi[vec_z_pos >= 0] - vec_random_dist_rphi[vec_z_pos >= 0]
        vec_fluctuation_dist_z = \
                vec_mean_dist_z[vec_z_pos >= 0] - vec_random_dist_z[vec_z_pos >= 0]
    elif selopt_output == 1:
        vec_fluctuation_dist_r = \
                vec_mean_dist_r[vec_z_pos < 0] - vec_random_dist_r[vec_z_pos < 0]
        vec_fluctuation_dist_rphi = \
                vec_mean_dist_rphi[vec_z_pos < 0] - vec_random_dist_rphi[vec_z_pos < 0]
        vec_fluctuation_dist_z = \
                vec_mean_dist_z[vec_z_pos < 0] - vec_random_dist_z[vec_z_pos < 0]
    elif selopt_output == 2:
        vec_fluctuation_dist_r = vec_mean_dist_r - vec_random_dist_r
        vec_fluctuation_dist_rphi = vec_mean_dist_rphi - vec_random_dist_rphi
        vec_fluctuation_dist_z = vec_mean_dist_z - vec_random_dist_z

    return [vec_mean_sc, vec_fluctuation_sc, vec_fluctuation_dist_r,
            vec_fluctuation_dist_rphi, vec_fluctuation_dist_z]


def load_train_apply(input_data, event_index, selopt_input, selopt_output,
                     grid_r, grid_rphi, grid_z, opt_train, opt_pred):

    [vec_mean_sc, vec_fluctuation_sc, vec_fluctuation_dist_r,
     vec_fluctuation_dist_rphi, vec_fluctuation_dist_z] = \
        load_data(input_data, event_index, selopt_input, selopt_output)
    dim_input = sum(opt_train)
    dim_output = sum(opt_pred)
    inputs = np.empty((grid_rphi, grid_r, grid_z, dim_input))
    exp_outputs = np.empty((grid_rphi, grid_r, grid_z, dim_output))

    indexfillx = 0 # TODO: Will it be used for something?
    # FIXME: These settings get overwritten - intentionally?
    if opt_train[0] == 1:
        inputs[:, :, :, indexfillx] = \
                vec_mean_sc.reshape(grid_rphi, grid_r, grid_z)
        indexfillx = indexfillx + 1
    if opt_train[1] == 1:
        inputs[:, :, :, indexfillx] = \
                vec_fluctuation_sc.reshape(grid_rphi, grid_r, grid_z)
        indexfillx = indexfillx + 1

    if sum(opt_pred) > 1:
        print("Multioutput not implemented yet!")
        return 0
    indexfilly = 0 # TODO: Will it be used for something?
    if opt_pred[0] == 1:
        exp_outputs[:, :, :, indexfilly] = \
                vec_fluctuation_dist_r.reshape(grid_rphi, grid_r, grid_z)
        indexfilly = indexfilly + 1
    if opt_pred[1] == 1:
        exp_outputs[:, :, :, indexfilly] = \
                vec_fluctuation_dist_rphi.reshape(grid_rphi, grid_r, grid_z)
        indexfilly = indexfilly + 1
    if opt_pred[2] == 1:
        exp_outputs[:, :, :, indexfilly] = \
                vec_fluctuation_dist_z.reshape(grid_rphi, grid_r, grid_z)
        indexfilly = indexfilly + 1
    #print("DIMENSION INPUT TRAINING", inputs.shape)
    #print("DIMENSION OUTPUT TRAINING", exp_outputs.shape)

    return inputs, exp_outputs


def get_event_mean_indices(maxrandomfiles_train, maxrandomfiles_apply, range_mean_index,
                           range_event_train, range_event_test, range_event_apply):
    all_indices_events_means_train = []
    all_indices_events_means_apply = []
    for imean in np.arange(range_mean_index[0], range_mean_index[1] + 1):
        for ievent in np.arange(maxrandomfiles_train):
            all_indices_events_means_train.append([ievent, imean])
        for ievent in np.arange(maxrandomfiles_apply):
            all_indices_events_means_apply.append([ievent, imean])

    sel_indices_events_means_train = random.sample(all_indices_events_means_train, \
        maxrandomfiles_train * (range_mean_index[1] + 1 - range_mean_index[0]))
    sel_indices_events_means_apply = random.sample(all_indices_events_means_apply, \
        maxrandomfiles_apply * (range_mean_index[1] + 1 - range_mean_index[0]))

    indices_events_means_train = [sel_indices_events_means_train[index] \
        for index in range(range_event_train[0], range_event_train[1])]
    indices_events_means_test = [sel_indices_events_means_train[index] \
        for index in range(range_event_test[0], range_event_test[1])]
    indices_events_means_apply = [sel_indices_events_means_apply[index] \
        for index in range(range_event_apply[0], range_event_apply[1])]

    partition = {'train': indices_events_means_train,
                 'validation': indices_events_means_test,
                 'apply': indices_events_means_apply}

    return sel_indices_events_means_train, partition
