from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from model.layer import *
from model.loss import std_mae, std_rmse


def arxiv_model(hyper):
    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]
    units_conv = hyper["units_conv"]
    units_dense = hyper["units_dense"]
    num_layers = hyper["num_layers"]
    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]
    pooling = hyper["pooling"]
    outputs = hyper["outputs"]
    normalize_adj = hyper["normalize_adj"]
    normalize_pos = hyper["normalize_pos"]

    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))

    sc, vc = GraphEmbed()([atoms, adjms, dists])
    ad, di = GraphNormalize(normalize_adj=normalize_adj, normalize_pos=normalize_pos)([adjms, dists])

    for _ in range(num_layers):
        sc_s = GraphSToS(units_conv, activation='relu')([sc, adjms])
        sc_v = GraphVToS(units_conv, activation='relu')([vc, di])

        vc_s = GraphSToV(units_conv, activation='relu')([sc, di])
        vc_v = GraphVToV(units_conv, activation='relu')([vc, adjms])

        sc = GraphConvS(units_conv, pooling=pooling, activation='relu')([sc_s, sc_v, ad])
        vc = GraphConvV(units_conv, pooling=pooling, activation='relu')([vc_s, vc_v, ad])

    out = GraphGatherSV(pooling=pooling)([sc, vc])
    out = Flatten()(out)
    out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(out)
    out = Dropout(0.5)(out)
    out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(out)

    if task == "regression":
        out = Dense(outputs, activation='linear', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(outputs, activation='sigmoid', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    elif task == "classification":
        out = Dense(outputs, activation='softmax', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    else:
        raise ValueError("Unsupported task on model generation.")

    return model

