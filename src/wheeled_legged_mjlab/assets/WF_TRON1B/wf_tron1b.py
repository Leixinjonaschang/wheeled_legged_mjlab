'''
The robot configuration is defined here, including:
1. actuators setting, 
2. articulation part (joint specifying) 
3. sensor specifying 
4. final entity defining with the above configures and the xml file 
'''

from pathlib import Path
import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

current_dir: Path = Path(__file__).parent.resolve()

WF_TRON1B_XML: Path = current_dir / "xml" / "robot.xml"
assert WF_TRON1B_XML.exists(), f"XML file not found: {WF_TRON1B_XML}"

# Parsing XML to MjSpec
def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(WF_TRON1B_XML))

# Leg actuation joint and actuation type defining
WF_TRON1B_LEG_ACTUATORS = XmlActuatorCfg(
    target_names_expr=("abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"),
    command_field="position",
)

# Wheel actuation joint and actuation type defining
WF_TRON1B_WHEEL_ACTUATORS = XmlActuatorCfg(
    target_names_expr=("wheel_[RL]_Joint",),
    command_field="velocity",
)

# Actuation final defining
WF_TRON1B_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        WF_TRON1B_LEG_ACTUATORS,
        WF_TRON1B_WHEEL_ACTUATORS,
    ),
)

# Contact sensor defining
WF_TRON1B_CONTACT_SENSOR = ContactSensorCfg(
    name="contact_sensors",
    primary=ContactMatch(mode="body", pattern="base_Link", entity="robot"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=10,
)

# Initial state defining
WF_TRON1B_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.8 + 0.166),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)

# Final robot entity defining
WF_TRON1B_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=WF_TRON1B_INIT_STATE,
    articulation=WF_TRON1B_ARTICULATION,
)

if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.scene import SceneCfg, Scene
    from mjlab.terrains import TerrainEntityCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": WF_TRON1B_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")

    viewer.launch(scene.compile())
