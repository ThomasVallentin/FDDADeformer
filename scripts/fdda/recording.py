from .common import (Subset, 
                     ProgressBar,
                     get_first_shape, 
                     get_node_obj, 
                     make_inputs_filename, 
                     make_outputs_filename,
                     strided_indices)
from .model import save_model_description

from maya import cmds
from maya import OpenMaya as om

import numpy as np

import json
import os


# TODO: Transform each sample into a main_joint local position

class SubsetRecord(object):
    EPSILON = 1e-1
    
    def __init__(self, subset : Subset):
        self.subset = subset
        self.rest_pose = None
        self.inputs = np.ndarray((0, 12 * len(subset.affecting_joints)))
        self.outputs = np.ndarray((0, 3 * len(subset.vertices)))
        
    def update_affecting_joints(self, joint, outputs):
        if joint in self.subset.affecting_joints:
            return
        
        if (np.abs(outputs[strided_indices(self.subset.vertices, 3)]) > self.EPSILON).any():
            self.subset.affecting_joints = np.append(self.subset.affecting_joints, joint)
    
    def update_rest_pose(self, rest_poses):
        self.rest_pose = rest_poses[self.subset.affecting_joints].flatten()
        
    def set_samples(self, inputs, outputs):
        self.inputs = inputs[:, strided_indices(self.subset.affecting_joints, 12)]
        self.outputs = outputs[:, strided_indices(self.subset.vertices, 3)]
        
        interesting_samples = (np.abs(self.inputs - self.rest_pose) > self.EPSILON).any(axis=1)
        self.inputs = self.inputs[interesting_samples]
        self.outputs = self.outputs[interesting_samples]
        
    def add_sample(self, inputs, outputs):
        np.append(self.inputs, 
                  [inputs[strided_indices(self.subset.affecting_joints, 12)]], 
                  axis=0)
        np.append(self.outputs, 
                  [outputs[strided_indices(self.subset.vertices, 3)]], 
                  axis=0)

    def __repr__(self):
        name = f'"{self.subset.main_joint}"' if self.subset else '<empty>'
        return f"{self.__class__.__name__}({name}, samples={self.inputs.shape[0]})"


class Recorder(object):
    TRANSLATE_LIMIT = (0, 0)  # only considering rotations for now
    ROTATE_LIMIT = (-90, 90) 

    def __init__(self, mesh):
        self.mesh = get_first_shape(mesh)
        self.linear_mesh = None
        self.datas = []
        self.sample_count = 0

    def initialize(self):
        skinclusters = cmds.ls(cmds.listHistory(self.mesh), type="skinCluster")
        if not skinclusters:
            raise RuntimeError(f"Could not find any skin cluster on {self.mesh}")
        elif len(skinclusters) > 1:
            raise RuntimeError(f"Find multiple skin clusters on {self.mesh}, "
                                "cannot define which to use as training base.")
        
        # Create the subsets using the hard skinning bindings
        self.datas = [SubsetRecord(subset) for subset in 
                      Subset.from_hard_skinning(self.mesh, skinclusters[0])]
        # We don't want to include the root joint so we assign its vertices to the next one
        self.datas[1].vertices = self.datas[0].vertices.append(self.datas[1].vertices)
        self.datas.pop(0)
        
        self.linear_mesh = "TRAINING_LINEAR_MESH"
        if not cmds.objExists(self.linear_mesh):
            self.linear_mesh = cmds.createNode("mesh", name=self.linear_mesh)
        cmds.connectAttr(f"{skinclusters[0]}.outputGeometry[0]", 
                         f"{self.linear_mesh}.inMesh",
                         force=True)

    def record(self, sample_count=100):
        if not self.datas:
            return
        
        self.sample_count = sample_count
        
        joints = self.all_joints()
        mesh_obj = get_node_obj(self.mesh)
        linear_mesh_obj = get_node_obj(self.linear_mesh)
        
        # Add a sample of the rest pose
        rest_inputs, rest_outputs = self.compute_sample(joints, mesh_obj, linear_mesh_obj)
        all_inputs = rest_inputs[np.newaxis, :]
        all_outputs = rest_outputs[np.newaxis, :]
        
        with ProgressBar.context("FDDADeformer : Sampling the joints...", 
                                 len(joints)) as context:

            for joint_index, joint in enumerate(joints):
                if context.is_cancelled():
                    context.stop()
                    break
                
                controller = joint.replace("skn", "ctrl")
                if not cmds.objExists(controller):
                    controller = joint.replace("skn", "FK_ctrl")
                    if not cmds.objExists(controller):
                        print(f"Could not find a controller for joint {joint}")
                        continue
                
                print(f"FDDADeformer : Sampling {controller}...")
                
                limits = self.get_limits(controller)
                for _ in range(sample_count):
                    # Moving a controller within its range of motion
                    translate = [self.sample_gaussian(mini, maxi) for mini, maxi in limits[:3]]
                    rotate = [self.sample_gaussian(mini, maxi) for mini, maxi in limits[3:]]
                    # cmds.xform(controller, os=True, a=True, t=translate)
                    cmds.xform(controller, os=True, ro=rotate)
                    
                    # Add this sample to all the subsets
                    inputs, outputs = self.compute_sample(joints, mesh_obj, linear_mesh_obj)
                    for data in self.datas:
                        data.update_affecting_joints(joint_index, outputs)
                        
                    all_inputs = np.append(all_inputs, inputs[np.newaxis, :], axis=0)
                    all_outputs = np.append(all_outputs, outputs[np.newaxis, :], axis=0)

                cmds.xform(controller, os=True, ro=(0,0,0))
                context.make_progress()

        # Dispatch the accumulated samples inside of each subset
        with ProgressBar.context("FDDADeformer : Updating the subsets using the sampled data...", 
                                 len(self.datas)) as context:
            for data in self.datas:
                if context.is_cancelled():
                    context.stop()
                    break
                
                # Update the rest pose of the subsets to allow them to discard non-relevant samples
                data.update_rest_pose(rest_inputs.reshape((len(joints), 12)))
                
                data.set_samples(all_inputs, all_outputs)
                context.make_progress()

    def finalize(self):
        if self.linear_mesh is not None:
            cmds.delete(cmds.listRelatives(self.linear_mesh, p=True))
            self.linear_mesh = None

    def all_joints(self):
        return [data.subset.main_joint for data in self.datas]

    @classmethod
    def compute_sample(cls, joints, mesh_obj, linear_mesh_obj):
        vertexCount = om.MFnMesh(mesh_obj).numVertices()
        
        inputs = np.empty((len(joints) * 12,))
        for i, joint in enumerate(joints):
            local_matrix = np.array(cmds.xform(joint, q=True, os=True, matrix=True))
            # Strip the fourth column since its value is constant and would just increase 
            # the amount of parameter that the network would have to learn
            input = local_matrix.reshape((4, 4))[:4, :3].reshape((12,))  
            inputs[i * 12: (i + 1) * 12] = input
            
        outputs = cls.get_displacement(linear_mesh_obj, mesh_obj).reshape((vertexCount * 3,))

        return inputs, outputs
        
    @classmethod
    def get_displacement(cls, linear_mesh, nonlinear_mesh):
        linear_fnmesh = om.MFnMesh(linear_mesh)
        nonlinear_fnmesh = om.MFnMesh(nonlinear_mesh)

        nonlinear_pnts = om.MPointArray()
        nonlinear_fnmesh.getPoints(nonlinear_pnts)
        
        linear_pnts = om.MPointArray()
        linear_fnmesh.getPoints(linear_pnts)

        vtx_count = nonlinear_pnts.length()
        result = np.empty((vtx_count * 3,))
        for i in range(vtx_count):
            displacement = nonlinear_pnts[i] - linear_pnts[i]
            result[i * 3] = displacement.x
            result[i * 3 + 1] = displacement.y
            result[i * 3 + 2] = displacement.z

        return result

    def save(self, directory, name=None):
        if not self.datas:
            return
        
        if name is None:
            name = self.mesh
        
        if not os.path.exists(directory):
            raise IOError(f"Could not find save directory {directory}")

        with ProgressBar.context("FDDADeformer : Updating the subsets using the sampled data...", 
                                 len(self.datas) + 1) as context:
                        
            for data in self.datas:
                np.savetxt(make_inputs_filename(name, data.subset.main_joint, directory), data.inputs)
                np.savetxt(make_outputs_filename(name, data.subset.main_joint, directory), data.outputs)
                
                context.make_progress()
        
            vertices_count = om.MFnMesh(get_node_obj(self.mesh)).numVertices()
            save_model_description(name, directory, self.mesh,
                                vertices_count, self.sample_count, 
                                [data.subset for data in self.datas])
            
            context.make_progress()
    
    def __repr__(self):
        return f"{self.__class__.__name__}(\"{self.mesh}\")"
   
    @classmethod
    def _get_limit(cls, value, mask, default):
        return tuple(v if m else d for v, m, d in zip(value, mask, default))

    @classmethod
    def get_limits(cls, transform):
        tx = cls._get_limit(cmds.transformLimits(transform, q=True, tx=True),
                            cmds.transformLimits(transform, q=True, etx=True),
                            cls.TRANSLATE_LIMIT)
        ty = cls._get_limit(cmds.transformLimits(transform, q=True, ty=True),
                            cmds.transformLimits(transform, q=True, ety=True),
                            cls.TRANSLATE_LIMIT)
        tz = cls._get_limit(cmds.transformLimits(transform, q=True, tz=True),
                            cmds.transformLimits(transform, q=True, etz=True),
                            cls.TRANSLATE_LIMIT)
        rx = cls._get_limit(cmds.transformLimits(transform, q=True, rx=True),
                            cmds.transformLimits(transform, q=True, erx=True),
                            cls.ROTATE_LIMIT)
        ry = cls._get_limit(cmds.transformLimits(transform, q=True, ry=True),
                            cmds.transformLimits(transform, q=True, ery=True),
                            cls.ROTATE_LIMIT)
        rz = cls._get_limit(cmds.transformLimits(transform, q=True, rz=True),
                            cmds.transformLimits(transform, q=True, erz=True),
                            cls.ROTATE_LIMIT)

        return tx, ty, tz, rx, ry, rz

    @classmethod
    def sample_gaussian(cls, mini, maxi):
        sample = mini - 1
        while sample < mini or sample > maxi:
            sample = np.random.normal([0.5 * (mini + maxi)], 1.5 * (maxi - mini))[0]
        
        return sample
