from fdda.common import ProgressBar, ProfilingScope, get_all_points
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
    parent_matrix_attr = None
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

        # input parent matrices
        matrix_attr = om.MFnMatrixAttribute()
        cls.parent_matrix_attr = matrix_attr.create("parentMatrix", "pm")
        matrix_attr.setArray(True)
        matrix_attr.setWritable(True)
        matrix_attr.setStorable(True)
        matrix_attr.setConnectable(True)
        matrix_attr.setHidden(False)
        
        cls.addAttribute(cls.parent_matrix_attr)
        cls.attributeAffects(cls.parent_matrix_attr, ompx.cvar.MPxGeometryFilter_outputGeom)

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
        
        inputs = np.zeros((expected_inputs_size, 12))
        for i in range(min(inputs_size, expected_inputs_size)):
            input_matrix_handle.jumpToElement(i)
            input_matrix = input_matrix_handle.inputValue().asMatrix()
            inputs[i] = np.array([input_matrix(r, c) for c in range(3) for r in range(4)])
        
        # Parent matrices
        parent_matrix_handle = datablock.inputArrayValue(self.parent_matrix_attr)
        parent_size = parent_matrix_handle.elementCount()
        
        parent_matrices = np.zeros((expected_inputs_size, 4, 4))
        for i in range(min(parent_size, expected_inputs_size)):
            parent_matrix_handle.jumpToElement(i)
            parent_matrix = parent_matrix_handle.inputValue().asMatrix()
            parent_matrices[i] = (np.array([parent_matrix(r, c) 
                                            for r in range(4) for c in range(4)])
                                  .reshape((4, 4))
                                  .transpose())
        
        outputs = np.zeros((vertices_count, 3))
        with ProfilingScope("Inference"):
            for i, (model, parent_matrix) in enumerate(zip(self._subset_models, parent_matrices)):
                subset_inputs = inputs[model.subset.affecting_joints]
                prediction = model.predict(subset_inputs.reshape((1, -1)))
                
                prediction = prediction.numpy().reshape((prediction.shape[1] // 3, 3))
                prediction = np.insert(prediction, 3, np.zeros(prediction.shape[0]), axis=1)
                
                # points_iter = ((point)[:3] for point in prediction)
                points_iter = (parent_matrix.dot(point)[:3] for point in prediction)
                outputs[model.subset.vertices] = np.fromiter(points_iter, 
                                                             count=prediction.shape[0], 
                                                             dtype=(prediction.dtype, 3))
        
        with ProfilingScope("Deformation"):
            positions = get_all_points(geom_it.allPositions)
            positions += outputs * envelope

            m_positions = om.MPointArray()
            for i, position in enumerate(positions):
                m_positions.append(*position.tolist())

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
