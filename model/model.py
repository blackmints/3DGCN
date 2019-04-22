from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, TimeDistributed, Concatenate, Add
from keras.optimizers import Adam
from keras.regularizers import l2
from model.layer import *
from model.loss import std_mae, std_rmse, masked_binary_crossentropy


def model_3DGCN(hyper):
    # Kipf adjacency, neighborhood mixing
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

    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))

    sc, vc = GraphEmbed()([atoms, dists])

    for _ in range(num_layers):
        sc_s = GraphSToS(units_conv, activation='relu')(sc)
        sc_v = GraphVToS(units_conv, activation='relu')([vc, dists])

        vc_s = GraphSToV(units_conv, activation='tanh')([sc, dists])
        vc_v = GraphVToV(units_conv, activation='tanh')(vc)

        sc = GraphConvS(units_conv, pooling='sum', activation='relu')([sc_s, sc_v, adjms])
        vc = GraphConvV(units_conv, pooling='sum', activation='tanh')([vc_s, vc_v, adjms])

    sc, vc = GraphGather(pooling=pooling)([sc, vc])
    sc_out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc)
    sc_out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc_out)

    vc_out = TimeDistributed(Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005)))(vc)
    vc_out = TimeDistributed(Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005)))(vc_out)
    vc_out = Flatten()(vc_out)

    out = Concatenate(axis=-1)([sc_out, vc_out])

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
