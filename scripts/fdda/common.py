from maya import cmds, mel
import maya.OpenMayaAnim as omanim
import maya.OpenMaya as om

import numpy as np

import os
import sys


class ProgressBar(object):
    def __init__(self, name=None):
        self._name = name
        if name is None:
            self._name = mel.eval('$tmp = $gMainProgressBar');

    def start(self, status, max_value=100):
        cmds.progressBar(self._name, 
                         edit=True,
                         beginProgress=True,
                         isInterruptable=True,
                         status=status,
                         maxValue=max_value)
    
    def stop(self):
        cmds.progressBar(self._name, 
                         edit=True, 
                         endProgress=True)

    def set_status(self, status):
        cmds.progressBar(self._name, 
                         edit=True,
                         status=status)

    def set_progress(self, progress):
        cmds.progressBar(self._name, 
                         edit=True,
                         progress=progress)

    def make_progress(self):
        cmds.progressBar(self._name, 
                         edit=True,
                         step=1)

    def is_cancelled(self):
        return cmds.progressBar(self._name, query=True, isCancelled=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    @classmethod
    def context(cls, status, max_value=None, name=None):
        instance = cls(name)
        instance.start(status, max_value)  
        return instance
    

class ProfilingScope(object):
    PROFILE_CATEGORY = om.MProfiler.addCategory("FDDADeformer")

    def __init__(self, name):
        self._args = [self.PROFILE_CATEGORY, 0, name]
        self._event_id = None

    def __enter__(self):
        self._event_id = om.MProfiler.eventBegin(*self._args)

    def __exit__(self, *args):
        om.MProfiler.eventEnd(self._event_id)
        

def get_first_shape(node_name):
    if cmds.ls(node_name, shapes=True):
        return node_name
    
    shapes = cmds.listRelatives(node_name, shapes=True)
    if shapes:
        return shapes[0]


def get_node_obj(node_name):
    result = om.MObject()
    sel = om.MSelectionList()
    sel.add(node_name)
    sel.getDependNode(0, result)

    return result


def get_node_dagpath(node_name):
    result = om.MDagPath()
    sel = om.MSelectionList()
    sel.add(node_name)
    sel.getDagPath(0, result)

    return result


def get_all_points(get_points_fn, flatten=False, **kwargs):
    points = om.MPointArray()
    get_points_fn(points, **kwargs)
    if flatten:
        return np.fromiter((points[i][axis]
                            for i in range(points.length()) 
                            for axis in range(3)), dtype=np.float32)
    
    return np.fromiter(([points[i][axis] for axis in range(3)]
                        for i in range(points.length())), 
                        dtype=np.dtype((np.float32, 3)))


def list_skin_influences(skincluster, fullPath=False):
    if isinstance(skincluster, str):
        skincluster = get_node_obj(skincluster)
        
    fn_skincluster = omanim.MFnSkinCluster(skincluster)

    # Get a list of the DagPaths of the joints affecting the mesh
    influences = om.MDagPathArray()
    fn_skincluster.influenceObjects(influences)
    get_name_func = om.MDagPath.fullPathName if fullPath else om.MDagPath.partialPathName
    return [get_name_func(influences[i]) for i in range(influences.length())]


def get_skin_weights(mesh_dagpath, skincluster, fullPath=False):
    if isinstance(skincluster, str):
        skincluster = get_node_obj(skincluster)
        
    # Get a list of the DagPaths of the joints affecting the mesh
    influences = list_skin_influences(skincluster, fullPath=fullPath)
    
    # Wrappers to C++ pointers to interact with the Maya API
    influence_count_util = om.MScriptUtil(len(influences))
    influence_count_ptr = influence_count_util.asUintPtr()
    influence_count = influence_count_util.asInt()

    weights = om.MDoubleArray()
    fn_skincluster = omanim.MFnSkinCluster(skincluster)
    fn_skincluster.getWeights(mesh_dagpath, om.MObject(), weights, influence_count_ptr)
    weights = np.fromiter((weights[i: i + influence_count] for i in range(0, weights.length(), influence_count )), 
                          dtype=np.dtype((float, influence_count)))
    
    return weights, influences


class Subset(object):
    def __init__(self, main_joint, vertices=None, affecting_joints=None):
        self.main_joint = main_joint
        self.vertices = vertices if vertices is not None else np.array([], dtype=np.int32)
        self.affecting_joints = affecting_joints if affecting_joints is not None else np.array([], dtype=np.int32)

    def name(self):
        return self.main_joint.split('|')[-1]

    def __str__(self):
        vertices = ", ".join([str(vtx) for vtx in sorted(self.vertices)])
        return f"{self.__class__.__name__}(\"{self.name()}\": {vertices})"
        
    def __repr__(self):
        vertices = ", ".join(str(vtx) for vtx in sorted(self.vertices))
        return f"{self.__class__.__name__}(\"{self.name()}\": {vertices})"
        
    def get_description(self):
        return {"main_joint": self.main_joint,
                "vertices": self.vertices.tolist(),
                "affecting_joints": self.affecting_joints.tolist()}
        
    @classmethod
    def validate_description(cls, description):
        return all(key in description 
                   for key in {"main_joint", "vertices", "affecting_joints"})
        
    @classmethod
    def from_description(cls, description):
        return cls(description["main_joint"], 
                   np.array(description["vertices"]), 
                   np.array(description["affecting_joints"]))

    @classmethod
    def from_hard_skinning(cls, mesh, skincluster):
        result = []
        mesh = get_first_shape(mesh)
        mesh_dagpath = get_node_dagpath(mesh)
        
        weights, influences = get_skin_weights(mesh_dagpath, skincluster)

        for joint_index, joint in enumerate(influences):
            result.append(Subset(joint, weights[:, joint_index].nonzero()[0]))
            
        return result
    

# Naming conventions

def make_inputs_filename(name, joint, directory=""):
    return os.path.join(directory, f"{name}_{joint}_inputs.csv")


def make_outputs_filename(name, joint, directory=""):
    return os.path.join(directory, f"{name}_{joint}_outputs.csv")


def make_model_filename(name, joint, directory=""):
    return os.path.join(directory, f"{name}_{joint}_Model.h5")


def make_model_description_filename(name, directory=""):
    return os.path.join(directory, f"{name}_model.json")


def make_inputs_mean_filename(name, joint, directory=""):
    return os.path.join(directory, f"{name}_{joint}_inputs_mean.csv")


def make_inputs_std_filename(name, joint, directory=""):
    return os.path.join(directory, f"{name}_{joint}_inputs_std.csv")


def make_outputs_mean_filename(name, joint, directory=""):
    return os.path.join(directory, f"{name}_{joint}_outputs_mean.csv")


def make_outputs_std_filename(name, joint, directory=""):
    return os.path.join(directory, f"{name}_{joint}_outputs_std.csv")


# Utils


def forget_modules(module_names=("fdda",)):
    for name in list(reversed(sys.modules)):
        for searched_name in module_names:
            if (name == searched_name or 
                name.startswith(searched_name + ".")):
                
                print(f"Forgetting module {name}")
                del sys.modules[name]
