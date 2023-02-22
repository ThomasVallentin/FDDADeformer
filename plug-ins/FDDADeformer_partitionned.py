from fdda.common import ProgressBar, ProfilingScope, strided_indices
from fdda.model import SubsetModel, validate_model_description

import maya.OpenMayaMPx as ompx
import maya.OpenMaya as om

import numpy as np
from keras.models import load_model

import json
import os


def normalize(data, mean, std):
    return (data - mean) / (std + np.finfo(np.float32).eps)


def denormalize(norm_data, mean, std):
    return norm_data * std + mean


class FDDADeformer(ompx.MPxGeometryFilter):
    TYPENAME = 'FDDADeformer'
    TYPEID = om.MTypeId(0x0003D53C)

    # Attributes declaration
    input_matrix_attr = None
    model_file_attr = None
    input_attr = ompx.cvar.MPxGeometryFilter_input
    input_geom_attr = ompx.cvar.MPxGeometryFilter_inputGeom
    output_geom_attr = ompx.cvar.MPxGeometryFilter_outputGeom
    envelope_attr = ompx.cvar.MPxGeometryFilter_envelope

    def __init__(self):
        super().__init__()

        self._subset_models = []
        self._vertices_count = -1
        self._model_file_is_dirty = True

    @classmethod
    def creator(cls):
        return cls()

    @classmethod
    def initialize(cls):
        # input file name
        default_filename = om.MFnStringData().create("")
        typed_attr = om.MFnTypedAttribute()
        cls.model_file_attr = typed_attr.create("modelFileName", "mfn", om.MFnData.kString, default_filename)
        typed_attr.setUsedAsFilename(True)
        typed_attr.setWritable(True)
        typed_attr.setStorable(True)
        typed_attr.setConnectable(True)
        typed_attr.setHidden(False)
        cls.addAttribute(cls.model_file_attr)
        cls.attributeAffects(cls.model_file_attr, ompx.cvar.MPxGeometryFilter_outputGeom)
        
        # input matrices
        matrix_attr = om.MFnMatrixAttribute()
        cls.input_matrix_attr = matrix_attr.create("inputMatrix", "im")
        matrix_attr.setArray(True)
        matrix_attr.setWritable(True)
        matrix_attr.setStorable(True)
        matrix_attr.setConnectable(True)
        matrix_attr.setHidden(False)
        
        cls.addAttribute(cls.input_matrix_attr)
        cls.attributeAffects(cls.input_matrix_attr, ompx.cvar.MPxGeometryFilter_outputGeom)

    def setDependentsDirty(self, dirtied_plug, affected_plugs):
        if dirtied_plug.attribute() == self.model_file_attr:
            self._model_file_is_dirty = True
            
        return super(FDDADeformer, self).setDependentsDirty(dirtied_plug, affected_plugs)

    def compute(self, plug, datablock):
        if self._model_file_is_dirty:
            # Get modelFileName plug and extract its data
            fn_node = om.MFnDependencyNode(self.thisMObject())
            model_file_plug = fn_node.findPlug(self.model_file_attr, True)
            fn_string_data = om.MFnStringData(model_file_plug.asMObject())
            raw_filename = fn_string_data.string()
            
            # Resolve the path using maya api
            model_file_object = om.MFileObject()
            model_file_object.setResolveMethod(om.MFileObject.kInputFile)
            model_file_object.setRawFullName(raw_filename)
            filepath = model_file_object.resolvedFullName()
            self._load_model_file(filepath)
        
        return super(FDDADeformer, self).compute(plug, datablock)

    def _load_model_file(self, filepath):
        self._model_file_is_dirty = False
        self._subset_models = []
        
        if not filepath:
            return
        
        try:
            with open(filepath, "r") as json_file:
                model_description = json.load(json_file)
                
            assert validate_model_description(model_description)

            self._vertices_count = model_description["vertices_count"]
            directory = os.path.dirname(os.path.normpath(filepath))

            subset_count = len(model_description["subsets"])
            message = (f"FDDADeformer : Reading and loading "
                        "models for {subset_count} subsets...")
            with ProgressBar.context(message, subset_count) as context:
                for subset_description in model_description["subsets"]:
                    if context.is_cancelled():
                        context.stop()
                        raise StopIteration

                    model = SubsetModel.from_description(subset_description, 
                                                         model_description["name"])
                    model.load(directory)
                    self._subset_models.append(model)

                    context.make_progress()
                
        except Exception as e:
            print (f"FDDADeformer : Invalid model description {filepath}, "
                    "nothing will happen.")
            self._subset_models = []
            self._vertices_count = -1
            self._model_file_is_dirty = True
            raise 
        
    def deform(self, datablock, geom_it, local_to_world_matrix, geom_index):
        if not self._subset_models:
            return
        
        # Input mesh
        input_geom_handle = datablock.outputArrayValue(self.input_attr)
        input_geom_handle.jumpToElement(geom_index)
        input_geom_obj = input_geom_handle.outputValue().child(self.input_geom_attr).asMesh()
        input_geom = om.MFnMesh(input_geom_obj)

        vertices_count = input_geom.numVertices()
        if vertices_count != self._vertices_count:
            print("FDDADeformer : "
                "Input mesh doesn't fit the model requirements "
                "({} vertices instead of {})."
                .format(vertices_count, self._vertices_count))

        # Enveloppe
        envelope = datablock.inputValue(self.envelope_attr).asFloat()
        
        # Input matrix
        input_matrix_handle = datablock.inputArrayValue(self.input_matrix_attr)
        inputs_size = input_matrix_handle.elementCount()
        
        expected_inputs_size = len(self._subset_models)
        inputs = np.zeros(expected_inputs_size * 12)
        
        for i in range(min(inputs_size, expected_inputs_size)):
            input_matrix_handle.jumpToElement(i)
            input_matrix = input_matrix_handle.inputValue().asMatrix()
            inputs[i * 12: (i + 1) * 12] = [input_matrix(r, c) 
                                            for r in range(4) for c in range(3)]
        
        outputs = np.zeros(vertices_count * 3)
        with ProfilingScope("Inference"):
            for i, model in enumerate(self._subset_models):
                subset_inputs = inputs[strided_indices(model.subset.affecting_joints, 12)]
                prediction = model.predict(subset_inputs[np.newaxis, :]).numpy().flatten()
                outputs[strided_indices(model.subset.vertices, 3)] = prediction

        with ProfilingScope("Deformation"):
            positions = om.MPointArray()
            geom_it.allPositions(positions)
            positions = np.array([positions[i][axis]
                                  for i in range(positions.length()) 
                                  for axis in range(3)])
            positions += outputs * envelope

            m_positions = om.MPointArray()
            for i in range(0, positions.shape[0], 3):
                m_positions.append(float(positions[i]),
                                   float(positions[i + 1]),
                                   float(positions[i + 2]))

            geom_it.setAllPositions(m_positions)
    

def initializePlugin(plugin_obj):
    plugin = ompx.MFnPlugin(plugin_obj, "Thomas Vallentin", "alpha_0.1")
    plugin.registerNode(FDDADeformer.TYPENAME,
                        FDDADeformer.TYPEID,
                        FDDADeformer.creator,
                        FDDADeformer.initialize,
                        ompx.MPxNode.kDeformerNode)


def uninitializePlugin(plugin_obj):
    plugin = ompx.MFnPlugin(plugin_obj)
    plugin.deregisterNode(FDDADeformer.TYPEID)
