import os
import time

import numpy as np
import tensorflow as tf
from keras.models import load_model

import maya.OpenMayaMPx as ompx
import maya.OpenMaya as om

TRAINING_FILES_DIR = r"D:\M2\DeepLearning\trained"

WEIGHTS_FILE = os.path.join(TRAINING_FILES_DIR, "Model.h5")
INPUTS_MEAN_FILE = os.path.join(TRAINING_FILES_DIR, "inputs_mean.csv")
INPUTS_STD_FILE = os.path.join(TRAINING_FILES_DIR, "inputs_std.csv")
OUTPUTS_MEAN_FILE = os.path.join(TRAINING_FILES_DIR, "outputs_mean.csv")
OUTPUTS_STD_FILE = os.path.join(TRAINING_FILES_DIR, "outputs_std.csv")

def normalize(data, mean, std):
    return (data - mean) / (std + np.finfo(np.float32).eps)

def denormalize(norm_data, mean, std):
    return norm_data * std + mean


class ProfilingScope(object):
    PROFILE_CATEGORY = om.MProfiler.addCategory("FDDADeformer")

    def __init__(self, name):
        """
        MProfilingScope (categoryId,
                         colorIndex (om.MProfiler.ProfilingColor),
                         eventName,
                         description=None,
                         associatedNode = om.MObject.kNullObj,
                         ReturnStatus = None)       
        """
        self._args = [self.PROFILE_CATEGORY, 0, name]
        self._event_id = None

    def __enter__(self):
        self._event_id = om.MProfiler.eventBegin(*self._args)

    def __exit__(self, *args):
        om.MProfiler.eventEnd(self._event_id)


class FDDADeformer(ompx.MPxGeometryFilter):
    TYPENAME = 'FDDADeformer'
    TYPEID = om.MTypeId(0x0003D53C)

    # Attributes declaration
    input_matrix = None
    input = ompx.cvar.MPxGeometryFilter_input
    input_geom = ompx.cvar.MPxGeometryFilter_inputGeom
    output_geom = ompx.cvar.MPxGeometryFilter_outputGeom
    envelope = ompx.cvar.MPxGeometryFilter_envelope
    group_id = ompx.cvar.MPxGeometryFilter_groupId

    def __init__(self):
        super().__init__()

        self._model = load_model(WEIGHTS_FILE)
        self._inputs_mean = np.loadtxt(INPUTS_MEAN_FILE)
        self._inputs_std = np.loadtxt(INPUTS_STD_FILE)
        self._outputs_mean = np.loadtxt(OUTPUTS_MEAN_FILE)
        self._outputs_std = np.loadtxt(OUTPUTS_STD_FILE)

    @classmethod
    def creator(cls):
        return cls()

    @classmethod
    def initialize(cls):
        # inputMatrix -> A transformer en array quand le réseau pourra apprendre les deformations de plusieurs joints
        input_matrix_attr = om.MFnMatrixAttribute()
        cls.input_matrix = input_matrix_attr.create('inputMatrix', 'im')
        input_matrix_attr.writable = True
        input_matrix_attr.storable = True
        input_matrix_attr.connectable = True
        input_matrix_attr.hidden = False
        
        cls.addAttribute(cls.input_matrix)
        cls.attributeAffects(cls.input_matrix, ompx.cvar.MPxGeometryFilter_outputGeom)

    def deform(self, datablock, geom_it, local_to_world_matrix, geom_index):
        # Input mesh
        input_handle = datablock.outputArrayValue(self.input)
        input_handle.jumpToElement(geom_index)
        input_geom_obj = input_handle.outputValue().child(self.input_geom).asMesh()
        mesh = om.MFnMesh(input_geom_obj)

        num_vertices = mesh.numVertices()
        model_num_vertices = self._model.output_shape[1] / 3
        if (num_vertices != model_num_vertices):
            raise RuntimeError("FDDADeformer : "
                "Input mesh doesn't fit the model requirements "
                "({} vertices instead of {})."
                .format(num_vertices, model_num_vertices))

        # Enveloppe
        envelope = datablock.inputValue(self.envelope).asFloat()
        
        # Input matrix
        input_matrix_handle = datablock.inputValue(self.input_matrix)
        input_matrix = input_matrix_handle.asMatrix()
        input_matrix = np.array([input_matrix(r, c) for r in range(3) for c in range(3)])
        input_matrix = normalize(input_matrix, self._inputs_mean, self._inputs_std)

        with ProfilingScope("Inference"):
            prediction = denormalize(self._model(input_matrix.reshape((1, 9))), 
                                     self._outputs_mean, 
                                     self._outputs_std)
            prediction = prediction.numpy().flatten()

        with ProfilingScope("Deformation"):
            positions = om.MPointArray()
            geom_it.allPositions(positions)
            positions = np.array([positions[i][axis]
                                  for i in range(positions.length()) 
                                  for axis in range(3)])
            positions += prediction * envelope

            m_positions = om.MPointArray()
            for i in range(0, positions.shape[0], 3):
                m_positions.append(float(positions[i]),
                                   float(positions[i + 1]),
                                   float(positions[i + 2]))

            geom_it.setAllPositions(m_positions)


class PrintFDDAStats(ompx.MPxCommand):
    NAME = "PrintFDDAStats"

    inference_time = 0.0
    inference_count = 0
    deform_time = 0.0
    deform_count = 0

    def doIt(self, args):
        print("Inference time: {}s\nDeform time: {}s"
              .format(self.inference_time / self.inference_count,
                      self.deform_time / self.deform_count))

    @classmethod
    def creator(cls):
        return PrintFDDAStats()


def initializePlugin(plugin_obj):
    plugin = ompx.MFnPlugin(plugin_obj, "Thomas Vallentin", "alpha_0.0")
    plugin.registerNode(FDDADeformer.TYPENAME,
                        FDDADeformer.TYPEID,
                        FDDADeformer.creator,
                        FDDADeformer.initialize,
                        ompx.MPxNode.kDeformerNode)

    plugin.registerCommand(PrintFDDAStats.NAME,
                           PrintFDDAStats.creator)


def uninitializePlugin(plugin_obj):
    plugin = ompx.MFnPlugin(plugin_obj)
    plugin.deregisterNode(FDDADeformer.TYPEID)
    plugin.deregisterCommand(PrintFDDAStats.NAME)
