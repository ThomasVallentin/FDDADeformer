from maya import cmds


PLUGIN_NAME = "FDDADeformer_partitionned"


def plugin_is_loaded():
    return cmds.pluginInfo(PLUGIN_NAME, q=True, loaded=True)


def load_plugin():
    return cmds.loadPlugin(PLUGIN_NAME)


def ensure_plugin_is_loaded():
    if not plugin_is_loaded():
        load_plugin()


def deform(mesh, joints, model_file=None):
    ensure_plugin_is_loaded()

    deformer = cmds.deformer(mesh, type="FDDADeformer")[0]
    for i, joint in enumerate(joints):
        if not cmds.objExists(joint):
            raise RuntimeError(f"Joint {joint} does not exist.")
        
        if not cmds.objectType(joint, isAType="transform"):
            raise TypeError(f"Joint {joint} is not a transform, "
                            "could not find a matrix and parentMatrix "
                            "to connect to the FDDADeformer.")
            
        cmds.connectAttr(f"{joint}.matrix", f"{deformer}.inputMatrix[{i}]")
        cmds.connectAttr(f"{joint}.worldMatrix[0]", f"{deformer}.parentMatrix[{i}]")

    if model_file is not None:
        cmds.setAttr(f"{deformer}.modelFileName", model_file, type="string")

    return deformer
