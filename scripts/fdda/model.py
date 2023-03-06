from .common import (Subset, 
                     make_inputs_filename, 
                     make_outputs_filename,
                     make_model_filename,
                     make_inputs_mean_filename, 
                     make_inputs_std_filename,
                     make_outputs_mean_filename,
                     make_outputs_std_filename)
import keras 
from keras import layers
from keras import regularizers
from keras.models import load_model
import numpy as np

from io import StringIO
import json
import os


# Data normalization

def normalize_for_training(raw_data):
    mean = np.mean(raw_data, axis=0)
    std = np.std(raw_data - mean, axis=0)
    norm_data = (raw_data - mean) / (std + np.finfo(np.float32).eps)

    return norm_data, mean, std


def normalize(raw_data, mean, std):
    return (raw_data - mean) / (std + np.finfo(np.float32).eps)


def denormalize(norm_data, mean, std):
    return norm_data * std + mean


# Model

def build_model(name, inputs_count, vertex_count, layer_size=512, activation="tanh"):
    model = keras.Sequential(name="FDDA_{}".format(name), 
                             layers=[
        layers.Dense(layer_size, input_dim=inputs_count, activation=activation),
        layers.Dense(layer_size, activation=activation), 
        layers.Dense(vertex_count, activation="linear")
    ])

    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam, loss="mse", metrics=["mae"])

    return model


class SubsetModel(object):
    def __init__(self, subset : Subset, name=""):
        self.name = name
        self.subset = subset
        
        self.inputs_mean = None
        self.inputs_std = None
        self.outputs_mean = None
        self.outputs_std = None
        
        self.model = None
        self.history = None
        
    @classmethod
    def from_description(cls, description, name=None):
        subset = Subset.from_description(description)
        return cls(subset, name=name)
        
    def train(self, inputs, outputs, epochs=200, callbacks=None, **kwargs):
        # Normalizing the data
        (inputs, 
         self.inputs_mean, 
         self.inputs_std) = normalize_for_training(inputs)
        (outputs, 
         self.outputs_mean, 
         self.outputs_std) = normalize_for_training(outputs)
        
        self.model = build_model(self.subset.main_joint, 
                                 inputs.shape[1], 
                                 outputs.shape[1],
                                 **kwargs)

        self.history = self.model.fit(inputs, outputs, 
                                      batch_size=64, 
                                      epochs=epochs, 
                                      validation_split=0.3, 
                                      callbacks=callbacks)
        
        return self.history
    
    def predict(self, inputs):
        inputs = normalize(inputs, self.inputs_mean, self.inputs_std)
        prediction = self.model(inputs)
        return denormalize(prediction, self.outputs_mean, self.outputs_std)

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        self.model.save(make_model_filename(self.name, self.subset.main_joint, directory))
        np.savetxt(make_inputs_mean_filename(self.name, self.subset.main_joint, directory), self.inputs_mean)
        np.savetxt(make_inputs_std_filename(self.name, self.subset.main_joint, directory), self.inputs_std)
        np.savetxt(make_outputs_mean_filename(self.name, self.subset.main_joint, directory), self.outputs_mean)
        np.savetxt(make_outputs_std_filename(self.name, self.subset.main_joint, directory), self.outputs_std)

        summary = []
        self.model.summary(print_fn=summary.append)
        with open(os.path.join(directory, f"{self.name}_{self.subset.main_joint}.json"), "w") as summary_file:
            json.dump({"summary": summary, "history": self.history.history}, summary_file, indent=4)

    def load(self, directory):
        self.model = load_model(make_model_filename(self.name, self.subset.main_joint, directory))
        self.inputs_mean = np.loadtxt(make_inputs_mean_filename(self.name, self.subset.main_joint, directory))
        self.inputs_std = np.loadtxt(make_inputs_std_filename(self.name, self.subset.main_joint, directory))
        self.outputs_mean = np.loadtxt(make_outputs_mean_filename(self.name, self.subset.main_joint, directory))
        self.outputs_std = np.loadtxt(make_outputs_std_filename(self.name, self.subset.main_joint, directory))

    def read_associated_recording(self, directory):
        inputs_file = make_inputs_filename(self.name, self.subset.main_joint, directory)
        outputs_file = make_outputs_filename(self.name, self.subset.main_joint, directory)
    
        return np.loadtxt(inputs_file), np.loadtxt(outputs_file)


def save_model_description(name, directory, mesh, vertices_count, sample_count, seed, subsets):
    subsets_list = []
    saved_data = {"name": name, 
                  "trained_mesh": mesh, 
                  "vertices_count": vertices_count,
                  "sample_count": sample_count,
                  "seed": seed,
                  "subsets": subsets_list}
    for subset in subsets:
        subsets_list.append(subset.get_description())

    with open(os.path.join(directory, f"{name}_model.json"), "w") as json_file:
        json.dump(saved_data, json_file, indent=4)


def validate_model_description(description):
    keys = {"name", "trained_mesh", "vertices_count", "sample_count", "seed", "subsets"}
    return (all(key in description for key in keys) and
            all(Subset.validate_description(subset) 
                for subset in description["subsets"]))
