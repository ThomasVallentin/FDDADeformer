from .common import (Subset, 
                     ProgressBar,
                     get_first_shape, 
                     get_node_obj, 
                     make_inputs_filename, 
                     make_outputs_filename)
from .model import save_model_description

from maya import cmds
from maya import OpenMaya as om

import numpy as np

from enum import Enum
import os


class RootJointPolicy(Enum):
    SKIP = 0
    KEEP = 1
    MERGE = 2


class SubsetRecord(object):
    EPSILON = 1e-1
    
    def __init__(self, subset : Subset):
        self.subset = subset
        self.rest_pose = None
        self.inputs = np.ndarray((0, len(subset.affecting_joints) * 16))
        self.outputs = np.ndarray((0, len(subset.vertices) * 3))
        
    def update_affecting_joints(self, joint, outputs):
        if joint in self.subset.affecting_joints:
            return
        
        if (np.abs(outputs[self.subset.vertices]) > self.EPSILON).any():
            self.subset.affecting_joints = np.append(self.subset.affecting_joints, joint)
    
    def update_rest_pose(self, rest_poses):
        self.rest_pose = rest_poses[self.subset.affecting_joints]
        
    def set_samples(self, inputs, outputs, world_inverses):
        self.inputs = inputs[:, self.subset.affecting_joints]
        self.outputs = outputs[:, self.subset.vertices]

        # Localize outputs to the main joint
        self.outputs = np.insert(self.outputs, 3, np.zeros(self.outputs.shape[:2]), axis=2)
        for sample, world_inverse in zip(self.outputs, world_inverses):
            points_iter = (world_inverse.dot(point) for point in sample)
            # points_iter = (point for point in sample)
            sample[:] = np.fromiter(points_iter, count=sample.shape[0], dtype=(sample.dtype, 4))
        self.outputs = self.outputs[:, :, :3]
        
        self.inputs = self.inputs.reshape((self.inputs.shape[0], -1))
        self.outputs = self.outputs.reshape((self.outputs.shape[0], -1))

    def __repr__(self):
        name = f'"{self.subset.main_joint}"' if self.subset else '<empty>'
        return f"{self.__class__.__name__}({name}, samples={self.inputs.shape[0]})"


class Recorder(object):
    TRANSLATE_LIMIT = (0, 0)  # only considering rotations for now
    ROTATE_LIMIT = (-90, 90) 

    def __init__(self, mesh):
        self.mesh = get_first_shape(mesh)
        self.linear_mesh = None
        self.records = []
        self.sample_count = 0
        self.seed = 0

    def initialize(self, root_joint_policy=RootJointPolicy.MERGE, seed=None):
        skinclusters = cmds.ls(cmds.listHistory(self.mesh), type="skinCluster")
        if not skinclusters:
            raise RuntimeError(f"Could not find any skin cluster on {self.mesh}")
        elif len(skinclusters) > 1:
            raise RuntimeError(f"Find multiple skin clusters on {self.mesh}, "
                                "cannot define which to use as training base.")
        
        # Create the subsets using the hard skinning bindings
        self.records = [SubsetRecord(subset) for subset in 
                        Subset.from_hard_skinning(self.mesh, skinclusters[0])]
        
        if root_joint_policy == RootJointPolicy.SKIP:
            self.records = self.records[1:]
        elif root_joint_policy == RootJointPolicy.MERGE:
            # Assign the vertices of the root joint to the next one
            self.records[1].subset.vertices = np.append(self.records[0].subset.vertices,
                                                        self.records[1].subset.vertices)
            self.records.pop(0)
        
        if seed is not None:
            self.seed = seed

    def compute_affecting_joints(self, samples=4):
        joints = self.all_joints()
        mesh_obj = get_node_obj(self.mesh)
        linear_mesh_obj = get_node_obj(self.linear_mesh)
        
        with ProgressBar.context("FDDADeformer : Computing the joints affecting the subsets...", 
                                 len(joints)) as context:
            for joint_index, joint in enumerate(joints):
                if context.is_cancelled():
                    context.stop()
                    raise StopIteration("Recording interrupted by user input.")

                controller = self.get_controller(joint)
                if not controller:
                    print(f"Could not find a controller for joint {joint}")
                    context.make_progress()
                    continue
                
                limits = self.get_limits(controller)
                
                for sample in range(samples):
                    rotate = [self.sample_gaussian(mini, maxi) for mini, maxi in limits[3:]]
                    cmds.xform(controller, os=True, ro=rotate)
                    
                    _, outputs, _ = self.compute_sample(joints, mesh_obj, linear_mesh_obj)

                    for record in self.records:
                        record.update_affecting_joints(joint_index, outputs)

                    cmds.xform(controller, os=True, ro=(0,0,0))
                    
                context.make_progress()
                    
    def record(self, sample_count=100):
        if not self.records:
            return
        
        np.random.seed(self.seed)
        self.sample_count = sample_count
        
        # Generate a mesh corresponding to the linear deformation
        self.linear_mesh = "TRAINING_LINEAR_MESH"
        if not cmds.objExists(self.linear_mesh):
            self.linear_mesh = cmds.createNode("mesh", name=self.linear_mesh)
        skinclusters = cmds.ls(cmds.listHistory(self.mesh), type="skinCluster")
        cmds.connectAttr(f"{skinclusters[0]}.outputGeometry[0]", 
                         f"{self.linear_mesh}.inMesh",
                         force=True)
        
        joints = self.all_joints()
        mesh_obj = get_node_obj(self.mesh)
        linear_mesh_obj = get_node_obj(self.linear_mesh)
        
        # Ensure the rig is in rest pose
        self.reset_controllers(joints)
        
        # Compute which joint has influence on which subset (this will determine the inputs size)
        self.compute_affecting_joints()
        
        # Add a sample of the rest pose
        rest_inputs, rest_outputs, world_inverse = self.compute_sample(joints, mesh_obj, linear_mesh_obj)
        all_inputs = rest_inputs[np.newaxis, :]
        all_outputs = rest_outputs[np.newaxis, :]
        all_world_inverse = world_inverse[np.newaxis, :]
        
        # Accumulate samples
        with ProgressBar.context("FDDADeformer : Sampling the joints...", 
                                 sample_count) as context:
            for i in range(sample_count):
                if context.is_cancelled():
                    self.reset_controllers(joints)
                    context.stop()
                    raise StopIteration("Recording interrupted by user input.")

                # Creating a random pose based on the controllers's limits
                for joint in joints:
                    controller = self.get_controller(joint)
                    if not controller:
                        continue
                    
                    # Moving a controller within its range of motion
                    limits = self.get_limits(controller)
                    rotate = [self.sample_gaussian(mini, maxi) for mini, maxi in limits[3:]]
                    # translate = [self.sample_gaussian(mini, maxi) for mini, maxi in limits[:3]]
                    
                    # cmds.xform(controller, os=True, a=True, t=translate)
                    cmds.xform(controller, os=True, ro=rotate)
                    
                # Add this sample to all the subsets
                inputs, outputs, world_inverse = self.compute_sample(joints, mesh_obj, linear_mesh_obj)

                all_inputs = np.append(all_inputs, inputs[np.newaxis, :], axis=0)
                all_outputs = np.append(all_outputs, outputs[np.newaxis, :], axis=0)
                all_world_inverse = np.append(all_world_inverse, world_inverse[np.newaxis, :], axis=0)
                
                context.make_progress()
        
        # Dispatch the accumulated samples inside of each subset
        with ProgressBar.context("FDDADeformer : Updating the subsets using the sampled record...", 
                                 len(self.records)) as context:
            for i, record in enumerate(self.records):
                if context.is_cancelled():
                    self.reset_controllers(joints)
                    context.stop()
                    raise StopIteration("Recording interrupted by user input.")
                
                # Update the rest pose of the subsets to allow them to discard non-relevant samples
                record.update_rest_pose(rest_inputs.reshape((len(joints), 4, 4)))
                
                record.set_samples(all_inputs, all_outputs, all_world_inverse[:, i])
                context.make_progress()

        # Cleanup: Reset the controller transformations
        self.reset_controllers(joints)
        
    def generate_gym(self, sample_count=100):
        if not self.records:
            return 
        
        np.random.seed(self.seed)
        joints = self.all_joints()

        # Ensure the rig is in rest pose
        self.reset_controllers(joints)
        
        # Generate posess
        start_time = cmds.currentTime(q=True)
        with ProgressBar.context("FDDADeformer : Sampling the joints...", 
                                 sample_count) as context:
            for i in range(sample_count):
                if context.is_cancelled():
                    self.reset_controllers(joints)
                    context.stop()
                    raise StopIteration("Recording interrupted by user input.")

                # Creating a random pose based on the controllers's limits
                for joint in joints:
                    controller = self.get_controller(joint)
                    if not controller:
                        continue
                    
                    # Moving a controller within its range of motion
                    limits = self.get_limits(controller)
                    rotate = [self.sample_gaussian(mini, maxi) for mini, maxi in limits[3:]]
                    # translate = [self.sample_gaussian(mini, maxi) for mini, maxi in limits[:3]]
                    
                    # cmds.xform(controller, os=True, a=True, t=translate)
                    cmds.xform(controller, os=True, ro=rotate)
                    cmds.setKeyframe(controller, t=start_time + i)
                    
                # Increment the time
                # cmds.currentTime(start_time + i)
                
                context.make_progress()
        
    @classmethod
    def reset_controllers(self, joints):
        for joint in joints:
            controller = self.get_controller(joint)
            if controller:
                cmds.xform(controller, os=True, ro=(0,0,0))

    def finalize(self):
        if self.linear_mesh is not None:
            cmds.delete(cmds.listRelatives(self.linear_mesh, p=True))
            self.linear_mesh = None

    def all_joints(self):
        return [record.subset.main_joint for record in self.records]

    @classmethod
    def get_controller(cls, joint):
        controller = joint.replace("skn", "ctrl")
        if not cmds.objExists(controller):
            controller = joint.replace("skn", "FK_ctrl")
            if not cmds.objExists(controller):
                return
            
        return controller

    @classmethod
    def compute_sample(cls, joints, mesh_obj, linear_mesh_obj):       
        inputs = np.empty((len(joints), 4, 4))
        world_inverses = np.empty((len(joints), 4, 4))
        
        for i, joint in enumerate(joints):
            local_matrix = np.array(cmds.xform(joint, q=True, os=True, matrix=True))
            # Strip the fourth column since its value is constant and would just increase 
            # the amount of parameter that the network would have to learn
            inputs[i] = local_matrix.reshape((4, 4)).transpose()
            
            world_matrix = (np.array(cmds.getAttr(f"{joint}.worldInverseMatrix[0]"))
                            .reshape((4, 4))
                            .transpose())
            world_inverses[i] = world_matrix
            
        outputs = cls.get_displacement(linear_mesh_obj, mesh_obj)

        return inputs, outputs, world_inverses
        
    @classmethod
    def get_displacement(cls, linear_mesh, nonlinear_mesh):
        linear_fnmesh = om.MFnMesh(linear_mesh)
        nonlinear_fnmesh = om.MFnMesh(nonlinear_mesh)

        nonlinear_pnts = om.MPointArray()
        nonlinear_fnmesh.getPoints(nonlinear_pnts)
        
        linear_pnts = om.MPointArray()
        linear_fnmesh.getPoints(linear_pnts)

        vtx_count = nonlinear_pnts.length()
        result = np.empty((vtx_count, 3))
        for i in range(vtx_count):
            displacement = nonlinear_pnts[i] - linear_pnts[i]
            result[i] = [displacement.x, displacement.y, displacement.z]
            
        return result

    def save(self, directory, name=None):
        if not self.records:
            return
        
        if name is None:
            name = self.mesh
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        with ProgressBar.context(f"FDDADeformer : Saving dataset to {directory}...", 
                                 len(self.records) + 1) as context:

            for record in self.records:
                np.savetxt(make_inputs_filename(name, record.subset.main_joint, directory), record.inputs)
                np.savetxt(make_outputs_filename(name, record.subset.main_joint, directory), record.outputs)
                
                context.make_progress()
        
            vertices_count = om.MFnMesh(get_node_obj(self.mesh)).numVertices()
            save_model_description(name, directory, self.mesh,
                                   vertices_count, self.sample_count, self.seed,
                                   [record.subset for record in self.records])
            
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
