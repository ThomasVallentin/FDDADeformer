from fdda.common import get_all_points

import maya.OpenMayaMPx as ompx
import maya.OpenMaya as om

import numpy as np

class DistanceToColor(ompx.MPxDeformerNode):
    TYPENAME = 'distanceToColor'
    TYPEID = om.MTypeId(0x0003D53D)

    # Attributes declaration
    ref_mesh_attr = None
    near_attr = None
    far_attr = None
    equal_color_attr = None
    near_color_attr = None
    far_color_attr = None
    input_attr = ompx.cvar.MPxGeometryFilter_input
    input_geom_attr = ompx.cvar.MPxGeometryFilter_inputGeom
    output_geom_attr = ompx.cvar.MPxGeometryFilter_outputGeom

    minimum_distance = 0.0
    maximum_distance = 1.0

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
        # Ref mesh
        typed_attr = om.MFnTypedAttribute()
        cls.ref_mesh_attr = typed_attr.create("refMesh", "rm", om.MFnData.kMesh)
        typed_attr.setWritable(True)
        typed_attr.setStorable(True)
        typed_attr.setConnectable(True)
        typed_attr.setHidden(False)
        
        cls.addAttribute(cls.ref_mesh_attr)
        cls.attributeAffects(cls.ref_mesh_attr, cls.output_geom_attr)

        # Near
        num_attr = om.MFnNumericAttribute()
        cls.near_attr = num_attr.create("near", "near", om.MFnNumericData.kFloat, 0.0)
        num_attr.setWritable(True)
        num_attr.setKeyable(True)
        num_attr.setStorable(True)
        num_attr.setConnectable(True)
        num_attr.setHidden(False)
        
        cls.addAttribute(cls.near_attr)
        cls.attributeAffects(cls.near_attr, cls.output_geom_attr)

        # Far
        num_attr = om.MFnNumericAttribute()
        cls.far_attr = num_attr.create("far", "far", om.MFnNumericData.kFloat, 1.0)
        num_attr.setWritable(True)
        num_attr.setKeyable(True)
        num_attr.setStorable(True)
        num_attr.setConnectable(True)
        num_attr.setHidden(False)
        
        cls.addAttribute(cls.far_attr)
        cls.attributeAffects(cls.far_attr, cls.output_geom_attr)

        # Equal color
        num_attr = om.MFnNumericAttribute()
        cls.equal_color_attr = num_attr.createColor("equalColor", "ec")
        num_attr.setDefault(0.0, 1.0, 0.0)
        num_attr.setWritable(True)
        num_attr.setKeyable(True)
        num_attr.setStorable(True)
        num_attr.setConnectable(True)
        num_attr.setHidden(False)
        
        cls.addAttribute(cls.equal_color_attr)
        cls.attributeAffects(cls.equal_color_attr, cls.output_geom_attr)

        # Near color
        num_attr = om.MFnNumericAttribute()
        cls.near_color_attr = num_attr.createColor("nearColor", "nc")
        num_attr.setDefault(0.0, 0.0, 1.0)
        num_attr.setWritable(True)
        num_attr.setKeyable(True)
        num_attr.setStorable(True)
        num_attr.setConnectable(True)
        num_attr.setHidden(False)
        
        cls.addAttribute(cls.near_color_attr)
        cls.attributeAffects(cls.near_color_attr, cls.output_geom_attr)

        # Far color
        num_attr = om.MFnNumericAttribute()
        cls.far_color_attr = num_attr.createColor("farColor", "fc")
        num_attr.setDefault(1.0, 0.0, 0.0)
        num_attr.setWritable(True)
        num_attr.setKeyable(True)
        num_attr.setStorable(True)
        num_attr.setConnectable(True)
        num_attr.setHidden(False)
        
        cls.addAttribute(cls.far_color_attr)
        cls.attributeAffects(cls.far_color_attr, cls.output_geom_attr)

    def postConstructor(self):
        self.setDeformationDetails(ompx.MPxDeformerNode.kDeformsColors)

    def deform(self, datablock, geom_it, local_to_world_matrix, geom_index):
        # Input mesh
        input_mesh_handle = datablock.outputArrayValue(self.input_attr)
        input_mesh_handle.jumpToElement(geom_index)
        input_mesh_obj = input_mesh_handle.outputValue().child(self.input_geom_attr).asMesh()
        input_mesh = om.MFnMesh(input_mesh_obj)

        # Ref mesh
        ref_mesh_obj = datablock.inputValue(self.ref_mesh_attr).asMesh()
        ref_mesh = om.MFnMesh(ref_mesh_obj)
        
        if ref_mesh.numVertices() != input_mesh.numVertices():
            return
        
        # Distance range
        near = datablock.inputValue(self.near_attr).asFloat()
        far = datablock.inputValue(self.far_attr).asFloat()
        distance_range = far - near
        
        # Colors
        equal_color = np.array(datablock.inputValue(self.equal_color_attr).asFloat3())
        near_color = np.array(datablock.inputValue(self.near_color_attr).asFloat3())
        far_color = np.array(datablock.inputValue(self.far_color_attr).asFloat3())
        
        # Output mesh
        output_mesh_handle = datablock.outputArrayValue(self.output_geom_attr)
        output_mesh_obj = output_mesh_handle.inputValue().asMesh()
        output_mesh = om.MFnMesh(output_mesh_obj)

        # Compute distance between the input and ref meshes
        input_points = get_all_points(input_mesh.getPoints)
        ref_points = get_all_points(ref_mesh.getPoints)
        distances = np.linalg.norm(input_points - ref_points, axis=1)

        # Blend the colors based on the distances and the near/far values
        blend_factors = np.minimum(np.maximum(distances - near, 0.0) / distance_range, 
                                   1.0)[:, np.newaxis]
        colors = far_color * blend_factors + near_color * (1.0 - blend_factors)
        colors[blend_factors[:, 0] < 1e-4] = equal_color
        
        # Convert to maya data and apply the colors to the mesh
        indices = om.MIntArray()
        mcolors = om.MColorArray()
        for i, color in enumerate(colors):
            mcolors.append(*color, 1)
            indices.append(i)
        output_mesh.setVertexColors(mcolors, indices)


def initializePlugin(plugin_obj):
    plugin = ompx.MFnPlugin(plugin_obj, "Thomas Vallentin", "1.0")
    plugin.registerNode(DistanceToColor.TYPENAME,
                        DistanceToColor.TYPEID,
                        DistanceToColor.creator,
                        DistanceToColor.initialize,
                        ompx.MPxNode.kDeformerNode)


def uninitializePlugin(plugin_obj):
    plugin = ompx.MFnPlugin(plugin_obj)
    plugin.deregisterNode(DistanceToColor.TYPEID)
