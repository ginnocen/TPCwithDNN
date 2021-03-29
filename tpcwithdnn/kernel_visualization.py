#pylint: disable=too-many-arguments, too-many-function-args
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from tpcwithdnn.symmetry_padding_3d import SymmetryPadding3d
from tpcwithdnn.data_loader import load_data_original


def get_full_charge_response(n_phi, n_r, n_z, model_files, input_dir, \
                             input_sc_fluc, input_sc_mean, output_pred):
    # load model
    json_file = open("%s.json" % model_files, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = \
        model_from_json(loaded_model_json, {'SymmetryPadding3d': SymmetryPadding3d})
    loaded_model.load_weights("%s.h5" % model_files)

    # define data frame column names
    dist_names = np.array(['R', 'Rphi', 'Z'])[np.array([output_pred[0], \
                                                        output_pred[1], \
                                                        output_pred[2]]) > 0]
    column_names = np.array(['phi', 'r', 'z',
                             'flucSC', 'meanSC'])
    for dist_name in dist_names:
        column_names = np.append(column_names, ['flucDist' + dist_name + 'Pred'])

    # load position arrays
    arr_r_pos, arr_phi_pos, arr_z_pos, \
    _, _, \
    _, _, \
    _, _, \
    _, _ = \
        load_data_original(input_dir, [0,0])
    arr_r_pos = arr_r_pos[arr_z_pos>0]
    arr_phi_pos = arr_phi_pos[arr_z_pos>0]
    arr_z_pos = arr_z_pos[arr_z_pos>0]

    # define input densities, create data frame and fill with prediction
    arr_fluc_sc = np.full(arr_r_pos.size, input_sc_fluc)
    arr_mean_sc = np.full(arr_r_pos.size, input_sc_mean)

    df = pd.DataFrame({column_names[0]: arr_phi_pos,
                        column_names[1]: arr_r_pos,
                        column_names[2]: arr_z_pos,
                        column_names[3]: arr_fluc_sc,
                        column_names[4]: arr_mean_sc})

    input_single = np.empty((1, n_phi, n_r, n_z, 2))
    input_single[0, :, :, :, 0] = arr_mean_sc.reshape(n_phi, n_r, n_z)
    input_single[0, :, :, :, 1] = arr_fluc_sc.reshape(n_phi, n_r, n_z)

    mat_fluc_dist_predict_group = loaded_model.predict(input_single)
    mat_fluc_dist_predict = np.empty((sum(output_pred), arr_r_pos.size))
    for index_dist in range(sum(output_pred)):
        mat_fluc_dist_predict[index_dist, :] = \
            mat_fluc_dist_predict_group[0, :, :, :, index_dist].flatten()
        df[column_names[5 + index_dist]] = mat_fluc_dist_predict[index_dist, :]

    return df


def get_line_charge_response(n_phi, n_r, n_z, model_files, input_dir, i_phi, i_r, \
                             input_sc_fluc, output_pred):
    # load model
    json_file = open("%s.json" % model_files, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = \
        model_from_json(loaded_model_json, {'SymmetryPadding3d': SymmetryPadding3d})
    loaded_model.load_weights("%s.h5" % model_files)

    # define data frame column names
    dist_names = np.array(['R', 'Rphi', 'Z'])[np.array([output_pred[0], \
                                                        output_pred[1], \
                                                        output_pred[2]]) > 0]
    column_names = np.array(['phi', 'r', 'z',
                             'flucSC', 'meanSC'])
    for distName in dist_names:
        column_names = np.append(column_names, ['flucDist' + distName + 'Pred'])

    # load position arrays
    arr_r_pos, arr_phi_pos, arr_z_pos, \
    arr_mean_sc, _, \
    _, _, \
    _, _, \
    _, _ = \
        load_data_original(input_dir, [0,0])
    arr_mean_sc = arr_mean_sc[arr_z_pos>0]
    arr_r_pos = arr_r_pos[arr_z_pos>0]
    arr_phi_pos = arr_phi_pos[arr_z_pos>0]
    arr_z_pos = arr_z_pos[arr_z_pos>0]

    # define input densities, create data frame and fill with prediction
    input_single = np.zeros((1, n_phi, n_r, n_z, 2))
    input_single[0, :, :, :, 0] = arr_mean_sc.reshape(n_phi, n_r, n_z)
    input_single[0, i_phi, i_r, :, 1] = input_sc_fluc
    arr_mean_sc = input_single[0, :, :, :, 0].flatten()
    arr_fluc_sc = input_single[0, :, :, :, 1].flatten()

    df = pd.DataFrame({column_names[0]: arr_phi_pos,
                        column_names[1]: arr_r_pos,
                        column_names[2]: arr_z_pos,
                        column_names[3]: arr_fluc_sc,
                        column_names[4]: arr_mean_sc})

    mat_fluc_dist_predict_group = loaded_model.predict(input_single)
    mat_fluc_dist_predict = np.empty((sum(output_pred), arr_r_pos.size))
    for index_dist in range(sum(output_pred)):
        mat_fluc_dist_predict[index_dist, :] = \
            mat_fluc_dist_predict_group[0, :, :, :, index_dist].flatten()
        df[column_names[5 + index_dist]] = mat_fluc_dist_predict[index_dist, :]

    return df


def get_point_charge_response(n_phi, n_r, n_z, model_files, input_dir, i_phi, i_r, i_z, \
                              input_sc_fluc, input_sc_mean, output_pred):
    # load model
    json_file = open("%s.json" % model_files, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = \
        model_from_json(loaded_model_json, {'SymmetryPadding3d': SymmetryPadding3d})
    loaded_model.load_weights("%s.h5" % model_files)

    # define data frame column names
    dist_names = np.array(['R', 'Rphi', 'Z'])[np.array([output_pred[0], \
                                                        output_pred[1], \
                                                        output_pred[2]]) > 0]
    column_names = np.array(['phi', 'r', 'z',
                             'flucSC', 'meanSC'])
    for distName in dist_names:
        column_names = np.append(column_names, ['flucDist' + distName + 'Pred'])

    # load position arrays
    arr_r_pos, arr_phi_pos, arr_z_pos, \
    _, _, \
    _, _, \
    _, _, \
    _, _ = \
        load_data_original(input_dir, [0,0])
    arr_r_pos = arr_r_pos[arr_z_pos>0]
    arr_phi_pos = arr_phi_pos[arr_z_pos>0]
    arr_z_pos = arr_z_pos[arr_z_pos>0]

    # define input densities, create data frame and fill with prediction
    input_single = np.zeros((1, n_phi, n_r, n_z, 2))
    input_single[0, i_phi, i_r, i_z, 0] = input_sc_mean
    input_single[0, i_phi, i_r, i_z, 1] = input_sc_fluc
    arr_mean_sc = input_single[0, :, :, :, 0].flatten()
    arr_fluc_sc = input_single[0, :, :, :, 1].flatten()

    df = pd.DataFrame({column_names[0]: arr_phi_pos,
                        column_names[1]: arr_r_pos,
                        column_names[2]: arr_z_pos,
                        column_names[3]: arr_fluc_sc,
                        column_names[4]: arr_mean_sc})

    mat_fluc_dist_predict_group = loaded_model.predict(input_single)
    mat_fluc_dist_predict = np.empty((sum(output_pred), arr_r_pos.size))
    for index_dist in range(sum(output_pred)):
        mat_fluc_dist_predict[index_dist, :] = \
            mat_fluc_dist_predict_group[0, :, :, :, index_dist].flatten()
        df[column_names[5 + index_dist]] = mat_fluc_dist_predict[index_dist, :]

    return df
