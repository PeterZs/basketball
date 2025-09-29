from typing import Callable, Optional, Union, List, Dict, Any, Sequence
import os
from isaacgym import gymapi, gymtorch
import torch

from utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply, expmap2quat

class Env(object):
    UP_AXIS = 2
    CHARACTER_MODEL = None
    CAMERA_POS= 0, -4.5, 2.0
    CAMERA_FOLLOWING = True

    def __init__(self,
        n_envs: int, fps: int=30, frameskip: int=2,
        episode_length: Optional[Union[Callable, int]] = 300,
        control_mode: str = "position",
        substeps: int = 2,
        compute_device: int = 0,
        graphics_device: Optional[int] = None,
        character_model: Optional[str] = None,
        **kwargs
    ):
        self.viewer = None
        assert(control_mode in ["position", "torque", "free"])
        self.frameskip = frameskip
        self.fps = fps
        self.step_time = 1./self.fps
        self.substeps = substeps
        self.control_mode = control_mode
        self.episode_length = episode_length
        self.device = torch.device(compute_device)
        self.camera_pos = self.CAMERA_POS
        self.camera_following = self.CAMERA_FOLLOWING
        if graphics_device is None:
            graphics_device = compute_device
        self.character_model = self.CHARACTER_MODEL if character_model is None else character_model
        if type(self.character_model) == str:
            self.character_model = [self.character_model]

        self.simulation_step = 0
        self.info = {}
        self.root_updated_actors, self.dof_updated_actors = [], []
        sim_params = self.setup_sim_params()
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
        self.add_ground()
        self.envs, self.actors, self.actuated_dofs = self.create_envs(n_envs)
        n_actors_per_env = self.gym.get_actor_count(self.envs[0])
        self.actor_ids = torch.arange(n_actors_per_env * len(self.envs), dtype=torch.int32, device=self.device).view(len(self.envs), -1)
        controllable_actors = []
        for i in range(self.gym.get_actor_count(self.envs[0])):
            dof = self.gym.get_actor_dof_count(self.envs[0], i)
            if dof > 0: controllable_actors.append(i)
        self.actor_ids_having_dofs = \
            n_actors_per_env * torch.arange(len(self.envs), dtype=torch.int32, device=self.device).unsqueeze(-1) + \
            torch.tensor(controllable_actors, dtype=torch.int32, device=self.device).unsqueeze(-2)
        self.setup_action_normalizer()
        self.create_tensors()

        self.gym.prepare_sim(self.sim)

        self.root_tensor.fill_(0)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
        self.joint_tensor.fill_(0)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.joint_tensor))
        self.refresh_tensors()
        self.train()
        self.viewer_pause = False
        self.viewer_advance = False
        tar_env = 0 #len(self.envs)//4 + int(len(self.envs)**0.5)//2
        base_pos = self.root_tensor[tar_env, 0, :3].cpu().detach()
        self.cam_target = gymapi.Vec3(*self.vector_up(1.0, [base_pos[0], base_pos[1], base_pos[2]]))

        self.done = torch.ones(len(self.envs), dtype=torch.bool, device=self.device)

        self.act_dim = self.action_scale.size(-1)
        self.ob_dim = self.observe().size(-1)
        self.rew_dim = self.reward().size(-1)

        for i in range(self.gym.get_actor_count(self.envs[0])):
            rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], i)
            print("Links", sorted(rigid_body.items(), key=lambda x:x[1]), len(rigid_body))
            dof = self.gym.get_actor_dof_dict(self.envs[0], i)
            print("Joints", sorted(dof.items(), key=lambda x:x[1]), len(dof))

    def __del__(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, "sim"):
            self.gym.destroy_sim(self.sim)

    def eval(self):
        self.training = False
        
    def train(self):
        self.training = True

    def vector_up(self, val: float, base_vector=None):
        if base_vector is None:
            base_vector = [0., 0., 0.]
        base_vector[self.UP_AXIS] = val
        return base_vector
    
    def setup_sim_params(self, physx_params=dict()):
        p = gymapi.SimParams()
        p.dt = self.step_time/self.frameskip
        p.substeps = self.substeps
        p.up_axis = gymapi.UP_AXIS_Z if self.UP_AXIS == 2 else gymapi.UP_AXIS_Y
        p.gravity = gymapi.Vec3(*self.vector_up(-9.81))
        p.num_client_threads = 0
        p.physx.num_threads = 4
        p.physx.solver_type = 1
        p.physx.num_position_iterations = 8
        p.physx.num_velocity_iterations = 0
        # p.physx.num_subscenes = 4  # works only for CPU 
        p.physx.contact_offset = 0.001
        p.physx.rest_offset = 0.0
        p.physx.bounce_threshold_velocity = 0.2
        p.physx.max_depenetration_velocity = 10.0
        p.physx.default_buffer_size_multiplier = 5.0
        p.physx.max_gpu_contact_pairs = 8*1024*1024
        # FIXME IsaacGym Pr4 will provide unreliable results when collecting from all substeps
        p.physx.contact_collection = \
            gymapi.ContactCollection(gymapi.ContactCollection.CC_LAST_SUBSTEP) 
        #gymapi.ContactCollection(gymapi.ContactCollection.CC_ALL_SUBSTEPS)
        for k, v in physx_params.items():
            setattr(p.physx, k, v)
        p.use_gpu_pipeline = True # force to enable GPU
        p.physx.use_gpu = True
        return p

    def add_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(*self.vector_up(1.0))
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.75
        self.gym.add_ground(self.sim, plane_params)

    def add_actor(self, env_handle: int, env_id: int, assets: Dict[str, int]):
        pass
    
    def register_asset(self) -> Dict[str, int]:
        return dict()
    
    def create_envs(self, n: int, start_height: float=0.89, actuate_all_dofs: bool=True, asset_options: Dict[str, Any]=dict()):
        if self.control_mode == "position":
            control_mode = gymapi.DOF_MODE_POS
        elif self.control_mode == "torque":
            control_mode = gymapi.DOF_MODE_EFFORT
        else:
            control_mode = gymapi.DOF_MODE_NONE

        envs, actors = [], []
        env_spacing = 3

        actor_assets = []
        controllable_dofs = []
        for character_model in self.character_model:
            asset_opt = gymapi.AssetOptions()
            asset_opt.angular_damping = 0.01
            asset_opt.max_angular_velocity = 100.0
            asset_opt.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
            for k, v in asset_options.items():
                setattr(asset_opt, k, v)
            asset = self.gym.load_asset(self.sim,
                os.path.abspath(os.path.dirname(character_model)),
                os.path.basename(character_model),
                asset_opt)
            actor_assets.append(asset)
            if actuate_all_dofs:
                controllable_dofs.append([i for i in range(self.gym.get_asset_dof_count(asset))])
            else:
                actuators = []
                for i in range(self.gym.get_asset_actuator_count(asset)):
                    name = self.gym.get_asset_actuator_joint_name(asset, i)
                    actuators.append(self.gym.find_asset_dof_index(asset, name))
                    if actuators[-1] == -1:
                        raise ValueError("Failed to find joint with name {}".format(name))
                controllable_dofs.append(sorted(actuators) if len(actuators) else [])

        spacing_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
        spacing_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        n_envs_per_row = int(n**0.5)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.vector_up(start_height))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        aux_assets = self.register_asset()

        total_rigids = sum([self.gym.get_asset_rigid_body_count(asset) for asset in actor_assets] + \
                           [self.gym.get_asset_rigid_body_count(asset) for asset in aux_assets.values()]) + 5
        total_shapes = sum([self.gym.get_asset_rigid_shape_count(asset) for asset in actor_assets] + \
                           [self.gym.get_asset_rigid_shape_count(asset) for asset in aux_assets.values()]) + 5
        
        actuated_dofs = []
        for env_id in range(n):
            env = self.gym.create_env(self.sim, spacing_lower, spacing_upper, n_envs_per_row)
            self.gym.begin_aggregate(env, total_rigids, total_shapes, True)
            for aid, (asset, dofs) in enumerate(zip(actor_assets, controllable_dofs)):
                actor = self.gym.create_actor(env, asset, start_pose, "actor{}_{}".format(env_id, aid), env_id, -1, 0)
                dof_prop = self.gym.get_asset_dof_properties(asset)
                for k in range(len(dof_prop)):
                    if k in dofs:
                        dof_prop[k]["driveMode"] = control_mode
                    else:
                        dof_prop[k]["driveMode"] = gymapi.DOF_MODE_NONE
                        dof_prop[k]["stiffness"] = 0
                        dof_prop[k]["damping"] = 0
                self.gym.set_actor_dof_properties(env, actor, dof_prop)
                if env_id == n-1:
                    actors.append(actor)
                    actuated_dofs.append(dofs)
            aux_actors = self.add_actor(env, env_id, aux_assets)
            # if env_id == n-1:
            #     actors.extend(aux_actors)
            self.gym.end_aggregate(env)
            envs.append(env)
        return envs, actors, actuated_dofs

    def render(self):
        tar_env = 0 #min(len(self.envs)-1, 1) #len(self.envs)//4 + int(len(self.envs)**0.5)//2
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        base_pos = self.root_tensor[tar_env, 0, :3].cpu().detach()
        cam_pos = gymapi.Vec3(*self.vector_up(self.camera_pos[2], 
            [base_pos[0]+self.camera_pos[0], base_pos[1]+self.camera_pos[1], base_pos[2]+self.camera_pos[1]]))
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "TOGGLE_CAMERA_FOLLOWING")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "TOGGLE_PAUSE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "SINGLE_STEP_ADVANCE")
    
    def update_viewer(self):
        self.gym.poll_viewer_events(self.viewer)
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == "QUIT" and event.value > 0:
                exit()
            if event.action == "TOGGLE_CAMERA_FOLLOWING" and event.value > 0:
                self.camera_following = not self.camera_following
            if event.action == "TOGGLE_PAUSE" and event.value > 0:
                self.viewer_pause = not self.viewer_pause
            if event.action == "SINGLE_STEP_ADVANCE" and event.value > 0:
                self.viewer_advance = not self.viewer_advance
        if self.camera_following: self.update_camera()
        self.gym.step_graphics(self.sim)
        self.gym.clear_lines(self.viewer)

    def update_camera(self):
        tar_env = 0 #min(len(self.envs)-1, 1) #len(self.envs)//4 + int(len(self.envs)**0.5)//2
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, self.envs[tar_env])
        dx, dy = cam_trans.p.x - self.cam_target.x, cam_trans.p.y - self.cam_target.y
        base_pos = self.root_tensor[tar_env, 0, :3].cpu().detach()
        cam_pos = gymapi.Vec3(base_pos[0]+dx, base_pos[1]+dy, cam_trans.p.z)
        self.cam_target = gymapi.Vec3(base_pos[0], base_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)

    def refresh_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def create_tensors(self):
        self.lifetime = torch.zeros(len(self.envs), dtype=torch.int64, device=self.device)

        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_tensor = gymtorch.wrap_tensor(root_tensor)
        self.root_tensor = root_tensor.view(len(self.envs), -1, 13)

        num_links = self.gym.get_env_rigid_body_count(self.envs[0])
        link_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        link_tensor = gymtorch.wrap_tensor(link_tensor)
        self.link_tensor = link_tensor.view(len(self.envs), num_links, -1)

        num_dof = self.gym.get_env_dof_count(self.envs[0])
        joint_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        joint_tensor = gymtorch.wrap_tensor(joint_tensor)
        self.joint_tensor = joint_tensor.view(len(self.envs), num_dof, -1)  # n_envs x n_dof x 2

        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self.contact_force_tensor = contact_force_tensor.view(len(self.envs), -1, 3)

        # sensor_force_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # sensor_force_tensor = gymtorch.wrap_tensor(sensor_force_tensor)
        # self.sensor_force_tensor = sensor_force_tensor.view(len(self.envs), -1, 3)

        if self.actuated_dofs.size(-1) == self.joint_tensor.size(1):
            self.action_tensor = None
        else:
            self.action_tensor = torch.zeros_like(self.joint_tensor[..., 0])

    def setup_action_normalizer(self):
        actuated_dof = []
        dof_cnts = 0
        action_lower, action_upper = [], []
        action_scale = []
        for i, dofs in zip(range(self.gym.get_actor_count(self.envs[0])), self.actuated_dofs):
            actor = self.gym.get_actor_handle(self.envs[0], i)
            dof_prop = self.gym.get_actor_dof_properties(self.envs[0], actor)
            if len(dof_prop) < 1: continue
            if self.control_mode == "torque":
                action_lower.extend([-dof_prop["effort"][j] for j in dofs])
                action_upper.extend([dof_prop["effort"][j] for j in dofs])
                action_scale.extend([1]*len(dofs))
            else: # self.control_mode == "position":
                action_lower.extend([min(dof_prop["lower"][j], dof_prop["upper"][j]) for j in dofs])
                action_upper.extend([max(dof_prop["lower"][j], dof_prop["upper"][j]) for j in dofs])
                action_scale.extend([2]*len(dofs))
            for j in dofs:
                actuated_dof.append(dof_cnts+j)
            dof_cnts += len(dof_prop)
        action_offset = 0.5 * np.add(action_upper, action_lower)
        action_scale *= 0.5 * np.subtract(action_upper, action_lower)
        self.action_offset = torch.tensor(action_offset, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(action_scale, dtype=torch.float32, device=self.device)
        self.actuated_dofs = torch.tensor(actuated_dof, dtype=torch.int64, device=self.device)

    def process_actions(self, actions):
        a = actions*self.action_scale + self.action_offset
        if self.action_tensor is None:
            return a
        self.action_tensor[:, self.actuated_dofs] = a
        return self.action_tensor

    def reset(self):
        self.lifetime.zero_()
        self.done.fill_(True)
        self.info.clear()
        self.info["lifetime"] = self.lifetime
        self.request_quit = False
        self.obs = None

        self.i = 0
        self.blender = []

    def reset_done(self):
        if not self.viewer_pause:
            env_ids = torch.nonzero(self.done).view(-1)
            if len(env_ids):
                self.reset_envs(env_ids)
                if len(env_ids) == len(self.envs) or self.obs is None:
                    self.obs = self.observe()
                else:
                    self.obs[env_ids] = self.observe(env_ids)
        return self.obs, self.info
    
    def reset_envs(self, env_ids):
        ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)

        self.root_tensor[env_ids] = ref_link_tensor[:, self.root_links]
        self.link_tensor[env_ids] = ref_link_tensor
        if self.action_tensor is None:
            self.joint_tensor[env_ids] = ref_joint_tensor
        else:
            self.joint_tensor[env_ids.unsqueeze(-1), self.actuated_dofs] = ref_joint_tensor
        
        self.root_updated_actors.append(self.actor_ids[env_ids].flatten())
        self.dof_updated_actors.append(self.actor_ids_having_dofs[env_ids].flatten())
        self.lifetime[env_ids] = 0

    def do_simulation(self):
        # root tensor inside isaacgym would be overwritten
        # when set_actor_root_state_tensor is called multiple times before doing simulation
        if self.root_updated_actors:
            actor_ids = torch.unique(torch.cat(self.root_updated_actors))
            if actor_ids.numel() == self.actor_ids.numel():
                self.gym.set_actor_root_state_tensor(self.sim,
                    gymtorch.unwrap_tensor(self.root_tensor)
                )
            else:
                self.gym.set_actor_root_state_tensor_indexed(self.sim,
                    gymtorch.unwrap_tensor(self.root_tensor),
                    gymtorch.unwrap_tensor(actor_ids), actor_ids.numel()
                )
            self.root_updated_actors.clear()
        if self.dof_updated_actors:
            actor_ids = torch.unique(torch.cat(self.dof_updated_actors))
            if actor_ids.numel() == self.actor_ids_having_dofs.numel():
                self.gym.set_dof_state_tensor(self.sim,
                    gymtorch.unwrap_tensor(self.joint_tensor)
                )
            else:
                self.gym.set_dof_state_tensor_indexed(self.sim,
                    gymtorch.unwrap_tensor(self.joint_tensor),
                    gymtorch.unwrap_tensor(actor_ids), actor_ids.numel()
                )
            self.dof_updated_actors.clear()
        for _ in range(self.frameskip):
            self.gym.simulate(self.sim)
        self.simulation_step += 1

    def step(self, actions):
        if not self.viewer_pause or self.viewer_advance:
            self.apply_actions(actions)
            self.do_simulation()
            self.refresh_tensors()
            
            self.lifetime += 1
            if self.viewer is not None:
                self.gym.fetch_results(self.sim, True)
                self.viewer_advance = False

        if self.viewer is not None:
            self.update_viewer()
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)    # sync to simulation dt

        rewards = self.reward()
        if self.viewer_pause:
            terminate = torch.zeros((len(self.envs),), dtype=torch.bool, device=self.device)
            self.done = terminate
        else:
            overtime = self.overtime_check()
            terminate = self.termination_check()                    # N
            self.done = torch.logical_or(overtime, terminate)

        self.info["terminate"] = terminate
        self.obs = self.observe()
        self.request_quit = False if self.viewer is None else self.gym.query_viewer_has_closed(self.viewer)
        return self.obs, rewards, self.done, self.info

    def apply_actions(self, actions):
        actions = self.process_actions(actions)
        if self.control_mode == "position":
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_position_target_tensor(self.sim, actions)
        elif self.control_mode == "torque":
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, actions)
        else:
            actions = torch.stack((actions, torch.zeros_like(actions)), -1)
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_state_tensor(self.sim, actions)

    def init_state(self, env_ids):
        raise NotImplementedError()
    
    def observe(self, env_ids=None):
        raise NotImplementedError()
    
    def overtime_check(self):
        if self.episode_length is None: return None
        if callable(self.episode_length):
            return self.lifetime >= self.episode_length(self.simulation_step).to(self.lifetime.device)
        return self.lifetime >= self.episode_length

    def termination_check(self):
        return torch.zeros(len(self.envs), dtype=torch.bool, device=self.device)

    def reward(self):
        return torch.ones((len(self.envs), 0), dtype=torch.float32, device=self.device)

class DiscriminatorConfig(object):
    def __init__(self,
        key_links: Optional[Union[List[List[str]], List[str]]]=None, ob_horizon: Optional[int]=None, 
        parent_link: Optional[Union[List[str], str]]=None, local_pos: Optional[Union[List[bool], bool]]=None,
        replay_speed: Optional[str]=None, motion_file: Optional[str]=None,
        weight:Optional[float]=None
    ):
        self.motion_file = motion_file
        self.key_links = key_links
        self.local_pos = local_pos
        self.parent_link = parent_link
        self.replay_speed = replay_speed
        self.ob_horizon = ob_horizon
        self.weight = weight

from ref_motion import ReferenceMotion
import numpy as np

class ICCGANHumanoid(Env):

    CHARACTER_MODEL = os.path.join("assets", "humanoid.xml")
    CONTACTABLE_LINKS = ["right_foot", "left_foot"]
    UP_AXIS = 2

    GOAL_DIM = 0
    GOAL_REWARD_WEIGHT = None
    ENABLE_GOAL_TIMER = False
    GOAL_TENSOR_DIM = None

    OB_HORIZON = 4
    KEY_LINKS = None    # All links
    PARENT_LINK = None  # root link


    def __init__(self, *args, motion_file: str, discriminators: Dict[str, DiscriminatorConfig]={}, **kwargs):
        contactable_links = kwargs.get("contactable_links", self.CONTACTABLE_LINKS)
        goal_reward_weight = kwargs.get("goal_reward_weight", self.GOAL_REWARD_WEIGHT)
        self.enable_goal_timer = kwargs.get("enable_goal_timer", self.ENABLE_GOAL_TIMER)
        self.goal_tensor_dim = kwargs.get("goal_tensor_dim", self.GOAL_TENSOR_DIM)
        self.ob_horizon = kwargs.get("ob_horizon", self.OB_HORIZON)
        self.key_links = kwargs.get("key_links", self.KEY_LINKS)
        self.parent_link = kwargs.get("parent_link", self.PARENT_LINK)
        super().__init__(*args, **kwargs)

        n_envs = len(self.envs)
        n_links = self.char_link_tensor.size(1)
        n_dofs = self.char_joint_tensor.size(1)

        if contactable_links is None:
            self.contactable_links = None
        elif contactable_links:
            contact = np.zeros((n_envs, n_links), dtype=bool)
            for link in contactable_links:
                lids = []
                for actor in self.actors:
                    lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, link)
                    if lid >= 0:
                        contact[:, lid] = True
                        lids.append(lid)
                assert len(lids) > 0, "Unrecognized contactable link {}".format(link)
            self.contactable_links = torch.tensor(contact).to(self.contact_force_tensor.device)
        else:
            self.contactable_links = False

        if goal_reward_weight is not None:
            reward_weights = torch.empty((len(self.envs), self.rew_dim), dtype=torch.float32, device=self.device)
            if not hasattr(goal_reward_weight, "__len__"):
                goal_reward_weight = [goal_reward_weight]
            assert self.rew_dim == len(goal_reward_weight), "{} vs {}".format(self.rew_dim, len(goal_reward_weight))
            for i, w in zip(range(self.rew_dim), goal_reward_weight):
                reward_weights[:, i] = w
        elif self.rew_dim:
            goal_reward_weight = []
            assert self.rew_dim == len(goal_reward_weight), "{} vs {}".format(self.rew_dim, len(goal_reward_weight)) 

        n_comp = len(discriminators) + self.rew_dim
        if n_comp > 1:
            self.reward_weights = torch.zeros((n_envs, n_comp), dtype=torch.float32, device=self.device)
            weights = [disc.weight for _, disc in discriminators.items() if disc.weight is not None]
            total_weights = sum(weights) if weights else 0
            assert(total_weights <= 1), "Discriminator weights must not be greater than 1."
            n_unassigned = len(discriminators) - len(weights)
            rem = 1 - total_weights
            for disc in discriminators.values():
                if disc.weight is None:
                    disc.weight = rem / n_unassigned
                elif n_unassigned == 0:
                    disc.weight /= total_weights
        else:
            self.reward_weights = None

        self.discriminators = dict()
        max_ob_horizon = self.ob_horizon+1
        for i, (id, config) in enumerate(discriminators.items()):
            assert config.key_links is None or all(type(_) == str for _ in config.key_links) or all(type(_)==list for _ in config.key_links)
            assert ((config.parent_link is None or type(config.parent_link) == str) and (config.key_links is None or all(type(_) == str for _ in config.key_links))) or \
                (type(config.parent_link) == list and all(type(_)==list for _ in config.key_links) and len(config.parent_link) == len(config.key_links))

            if config.key_links is None:
                config.key_links = [None]
            elif all(type(_) == str for _ in config.key_links):
                config.key_links = [config.key_links]
            key_links = []
            for k in config.key_links:
                if k is None:
                    key_links.append([-1])
                else:
                    links = []
                    for link in k:
                        for actor in self.actors:
                            lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, link)
                            if lid != -1:
                                links.append(lid)
                                break
                        assert lid != -1, "Unrecognized key link {}".format(link)
                    links = sorted(links)
                    key_links.append(links)
            if type(config.parent_link) != list:
                config.parent_link = [config.parent_link]
            parent_link = []
            for link in config.parent_link:
                if link is None:
                    parent_link.append(-1)
                else:
                    for j in self.actors:
                        pid = self.gym.find_actor_rigid_body_handle(self.envs[0], j, link)
                        if pid != -1: break
                    assert pid != -1, "Unrecognized parent link {}".format(link)
                    parent_link.append(pid)

            config.parent_link = parent_link
            config.key_links = key_links
            
            if config.motion_file is None:
                config.motion_file = motion_file
            if config.ob_horizon is None:
                config.ob_horizon = self.ob_horizon+1
            config.id = i
            config.name = id
            self.discriminators[id] = config
            if self.reward_weights is not None:
                self.reward_weights[:, i] = config.weight
            max_ob_horizon = max(max_ob_horizon, config.ob_horizon)

        if max_ob_horizon != self.state_hist.size(0):
            self.state_hist = torch.zeros((max_ob_horizon, *self.state_hist.shape[1:]),
                dtype=self.root_tensor.dtype, device=self.device)
        if self.reward_weights is None:
            self.reward_weights = torch.ones((n_envs, 1), dtype=torch.float32, device=self.device)
        elif self.rew_dim > 0:
            if self.rew_dim > 1:
                self.reward_weights *= (1-reward_weights.sum(dim=-1, keepdim=True))
            else:
                self.reward_weights *= (1-reward_weights)
            self.reward_weights[:, -self.rew_dim:] = reward_weights
        

        self.link_state_hist = self.state_hist.view(self.state_hist.size(0), len(self.envs), -1, 13)
        self.info["ob_seq_lens"] = torch.zeros_like(self.lifetime)  # dummy result
        self.goal_dim = self.GOAL_DIM
        if isinstance(self.goal_dim, int):
            self.goal_dim = self.goal_dim, self.goal_dim
        else:
            assert isinstance(self.goal_dim[0], int) and isinstance(self.goal_dim[1], int)
        g = max(self.goal_dim)
        self.state_dim = (self.ob_dim-g)//self.ob_horizon
        if self.discriminators:
            self.info["disc_obs"] = self.observe_disc(self.state_hist)  # dummy result
            self.info["disc_obs_expert"] = self.info["disc_obs"]        # dummy result
            self.disc_dim = {
                name: ob.size(-1)
                for name, ob in self.info["disc_obs"].items()
            }
        else:
            self.disc_dim = {}

        self.ref_motion = self.build_motion_lib(motion_file)
        self.sampling_workers = []
        self.real_samples = []

    def build_motion_lib(self, motion_file: Union[str, Sequence[str]]):
        n_links = self.char_link_tensor.size(1)
        return ReferenceMotion(motion_file=motion_file, character_model=self.character_model,
            key_links=np.arange(n_links), device=self.device)
        
    def __del__(self):
        if hasattr(self, "sampling_workers"):
            for p in self.sampling_workers:
                p.terminate()
            for p in self.sampling_workers:
                p.join()
        super().__del__()

    def reset_done(self):
        obs, info = super().reset_done()
        info["ob_seq_lens"] = self.ob_seq_lens
        info["reward_weights"] = self.reward_weights
        return obs, info
    
    def reset(self):
        if self.goal_tensor is not None:
            self.goal_tensor.zero_()
            if self.goal_timer is not None: self.goal_timer.zero_()
        super().reset()

    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_goal(env_ids)
        
    def reset_goal(self, env_ids):
        pass
    
    def step(self, actions):
        obs, rews, dones, info = super().step(actions)
        if self.discriminators and self.training:
            info["disc_obs"] = self.observe_disc(self.state_hist)
            info["disc_obs_expert"] = self.fetch_real_samples()
        return obs, rews, dones, info

    def overtime_check(self):
        if self.goal_timer is not None:
            self.goal_timer -= 1
            env_ids = torch.nonzero(self.goal_timer <= 0).view(-1)
            if len(env_ids) > 0: self.reset_goal(env_ids)
        return super().overtime_check()

    def termination_check(self):
        if self.contactable_links is None:
            return torch.zeros_like(self.done)
        masked_contact = self.char_contact_force_tensor.clone()
        if self.contactable_links is not False:
            masked_contact[self.contactable_links] = 0          # N x n_links x 3

        contacted = torch.any(masked_contact.abs_() > 1., dim=-1)      # N x n_links
        too_low = self.link_pos[..., self.UP_AXIS] < 0.3    # N x n_links

        terminate = torch.any(torch.logical_and(contacted, too_low), -1)    # N x
        terminate *= (self.lifetime > 1)
        return terminate

    def init_state(self, env_ids):
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)
        return ref_link_tensor, ref_joint_tensor
    
    def create_tensors(self):
        super().create_tensors()
        n_dofs = sum([self.gym.get_actor_dof_count(self.envs[0], actor) for actor in self.actors])
        n_links = sum([self.gym.get_actor_rigid_body_count(self.envs[0], actor) for actor in self.actors])
        self.root_pos, self.root_orient = self.root_tensor[:, 0, :3], self.root_tensor[:, 0, 3:7]
        self.root_lin_vel, self.root_ang_vel = self.root_tensor[:, 0, 7:10], self.root_tensor[:, 0, 10:13]
        self.char_root_tensor = self.root_tensor[:, 0]
        if self.link_tensor.size(1) > n_links:
            self.link_pos, self.link_orient = self.link_tensor[:, :n_links, :3], self.link_tensor[:, :n_links, 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[:, :n_links, 7:10], self.link_tensor[:, :n_links, 10:13]
            self.char_link_tensor = self.link_tensor[:, :n_links]
        else:
            self.link_pos, self.link_orient = self.link_tensor[..., :3], self.link_tensor[..., 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[..., 7:10], self.link_tensor[..., 10:13]
            self.char_link_tensor = self.link_tensor
        if self.joint_tensor.size(1) > n_dofs:
            self.joint_pos, self.joint_vel = self.joint_tensor[:, :n_dofs, 0], self.joint_tensor[:, :n_dofs, 1]
            self.char_joint_tensor = self.joint_tensor[:, :n_dofs]
        else:
            self.joint_pos, self.joint_vel = self.joint_tensor[..., 0], self.joint_tensor[..., 1]
            self.char_joint_tensor = self.joint_tensor
        
        self.char_contact_force_tensor = self.contact_force_tensor[:, :n_links]
    
        self.state_hist = torch.zeros((self.ob_horizon+1, len(self.envs), n_links*13),
            dtype=self.root_tensor.dtype, device=self.device)

        assert self.key_links is None or all(type(_) == str for _ in self.key_links) or all(type(_)==list for _ in self.key_links)
        assert ((self.parent_link is None or type(self.parent_link) == str) and (self.key_links is None or all(type(_) == str for _ in self.key_links))) or \
            (type(self.parent_link) == list and all(type(_)==list for _ in self.key_links) and len(self.parent_link) == len(self.key_links))

        if self.key_links is None:
            self.key_links = [None]
        elif all(type(_) == str for _ in self.key_links):
            self.key_links = [self.key_links]
        key_links = []
        for k in self.key_links:
            if k is None:
                key_links.append([-1])
            else:
                links = []
                for link in k:
                    for actor in self.actors:
                        lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, link)
                        if lid != -1:
                            links.append(lid)
                            break
                    assert lid != -1, "Unrecognized key link {}".format(link)
                links = sorted(links)
                key_links.append(links)
        if type(self.parent_link) != list:
            self.parent_link = [self.parent_link]
        parent_link = []
        for link in self.parent_link:
            if link is None:
                parent_link.append(-1)
            else:
                for j in self.actors:
                    pid = self.gym.find_actor_rigid_body_handle(self.envs[0], j, link)
                    if pid != -1: break
                assert pid != -1, "Unrecognized parent link {}".format(link)
                parent_link.append(pid)
        self.key_links = key_links
        self.parent_link = parent_link
                
        if self.goal_tensor_dim:
            try:
                self.goal_tensor = [
                    torch.zeros((len(self.envs), dim), dtype=self.root_tensor.dtype, device=self.device)
                    for dim in self.goal_tensor_dim
                ]
            except TypeError:
                self.goal_tensor = torch.zeros((len(self.envs), self.goal_tensor_dim), dtype=self.root_tensor.dtype, device=self.device)
        else:
            self.goal_tensor = None
        self.goal_timer = torch.zeros((len(self.envs), ), dtype=torch.int32, device=self.device) if self.enable_goal_timer else None

        self.root_links = list(np.cumsum([0] + [self.gym.get_actor_rigid_body_count(self.envs[0], actor) for actor in range(self.gym.get_actor_count(self.envs[0]))])[:-1])


    def observe(self, env_ids=None):
        self.ob_seq_lens = self.lifetime+1
        n_envs = len(self.envs)
        if env_ids is None or len(env_ids) == n_envs:
            if not self.viewer or not self.viewer_pause:
                self.state_hist[:-1] = self.state_hist[1:].clone()
                self.state_hist[-1] = self.char_link_tensor.view(n_envs, -1)
            env_ids = None
        elif not self.viewer or not self.viewer_pause:
            self.state_hist[:-1, env_ids] = self.state_hist[1:, env_ids].clone()
            self.state_hist[-1, env_ids] = self.char_link_tensor[env_ids].view(len(env_ids), -1)
        return self._observe(env_ids)
    
    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens, self.key_links, self.parent_link,
            ).flatten(start_dim=1)
        else:
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:, env_ids], self.ob_seq_lens[env_ids], self.key_links, self.parent_link,
            ).flatten(start_dim=1)

    def observe_disc(self, state):
        seq_len = self.info["ob_seq_lens"]+1
        res = dict()
        for id, disc in self.discriminators.items():
            res[id] = observe_iccgan(state[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link,
                include_velocity=False, local_pos=disc.local_pos)
        return res

    def fetch_real_samples(self):
        if not self.real_samples:
            if not self.sampling_workers:
                self.disc_ref_motion = {}
                import torch.multiprocessing as mp
                mp.set_start_method("spawn")
                manager = mp.Manager()
                seed = np.random.get_state()[1][0]
                for n, config in self.discriminators.items():
                    q = manager.Queue(maxsize=1)
                    self.disc_ref_motion[n] = q

                    key_links = []
                    for k in config.key_links:
                        if k[0] == -1:
                            key_links = None
                            break
                        else:
                            key_links.extend(k)
                    if key_links is None:
                        key_links_index = config.key_links
                        parent_link_index = config.parent_link
                    else:
                        for p in config.parent_link:
                            if p == -1:
                                key_links.append(0)
                            else:
                                key_links.append(p)
                        key_links = sorted(list(set(key_links)))
                        key_links_index = []
                        for k in config.key_links:
                            if k[0] == -1:
                                key_links_index.append([-1])
                                continue
                            key_links_index.append([])
                            for _ in k:
                                key_links_index[-1].append(key_links.index(_))
                        parent_link_index = []
                        for p in config.parent_link:
                            if p == -1:
                                parent_link_index.append(-1)
                            else:
                                parent_link_index.append(key_links.index(p))
                    p = mp.Process(target=self.__class__.ref_motion_sample, args=(q,
                        seed+1+config.id, self.step_time, len(self.envs), config.ob_horizon, key_links_index, parent_link_index, config.local_pos, config.replay_speed,
                        dict(motion_file=config.motion_file, character_model=self.character_model,
                            key_links=key_links, device=self.device
                        )
                    ))
                    p.start()
                    self.sampling_workers.append(p)

            self.real_samples = [{n: None for n in self.disc_ref_motion.keys()} for _ in range(128)]
            for n, q in self.disc_ref_motion.items():
                for i, v in enumerate(q.get()):
                    self.real_samples[i][n] = v.to(self.device)
        return self.real_samples.pop()

    @staticmethod
    def ref_motion_sample(queue, seed, step_time, n_inst, ob_horizon, key_links, parent_link, local_pos, replay_speed, kwargs):
        np.random.seed(seed)
        torch.set_num_threads(1)
        lib = ReferenceMotion(**kwargs)
        if replay_speed is not None:
            replay_speed = eval(replay_speed)
        while True:
            obs = []
            for _ in range(128):
                if replay_speed is None:
                    dt = step_time
                else:
                    dt = step_time * replay_speed(n_inst)
                motion_ids, motion_times0 = lib.sample(n_inst, truncate_time=dt*(ob_horizon-1))
                motion_ids = np.tile(motion_ids, ob_horizon)
                motion_times = np.concatenate((motion_times0, *[motion_times0+dt*i for i in range(1, ob_horizon)]))
                link_tensor = lib.state(motion_ids, motion_times, with_joint_tensor=False)
                samples = link_tensor.view(ob_horizon, n_inst, -1)
                ob = observe_iccgan(samples, None, key_links, parent_link, include_velocity=False, local_pos=local_pos)
                obs.append(ob.cpu())
            queue.put(obs)

@torch.jit.script
def observe_iccgan(state_hist: torch.Tensor, seq_len: Union[torch.Tensor, None],
    key_links: List[List[int]], parent_link: List[int],
    include_velocity: bool=True, local_pos: Optional[bool]=None, ground_height:Optional[torch.Tensor]=None
):
    # state_hist: L x N x (1+N_links) x 13

    UP_AXIS = 2
    n_hist = state_hist.size(0)
    n_inst = state_hist.size(1)

    link_tensor = state_hist.view(n_hist, n_inst, -1, 13)

    obs = []
    for k, p in zip(key_links, parent_link):

        if k[0] == -1:
            link_pos, link_orient = link_tensor[...,:3], link_tensor[...,3:7]
        else:
            link_pos, link_orient = link_tensor[:,:,k,:3], link_tensor[:,:,k,3:7]

        if p == -1:
            root_tensor = state_hist[..., :13]
            if local_pos is True:
                origin = root_tensor[:,:, :3]          # L x N x 3
                orient = root_tensor[:,:,3:7]          # L x N x 4
            else:
                origin = root_tensor[-1,:, :3]          # N x 3
                orient = root_tensor[-1,:,3:7]          # N x 4

            heading = heading_zup(orient)               # (L x) N
            up_dir = torch.zeros_like(origin)
            up_dir[..., UP_AXIS] = 1                    # (L x) N x 3
            orient_inv = axang2quat(up_dir, -heading)   # (L x) N x 4
            orient_inv = orient_inv.view(-1, n_inst, 1, 4)   # L x N x 1 x 4 or 1 x N x 1 x 4

            origin = origin.clone()
            if ground_height is None:
                origin[..., UP_AXIS] = 0                # (L x) N x 3
            else:
                origin[..., UP_AXIS] = ground_height    # (L x) N x 3
            origin.unsqueeze_(-2)                       # (L x) N x 1 x 3
        else:
            if local_pos is True or local_pos is None:
                origin = link_tensor[:,:, p, :3]  # L x N x 3
                orient = link_tensor[:,:, p,3:7]  # L x N x 4
            else:
                origin = link_tensor[-1,:, p, :3]  # N x 3
                orient = link_tensor[-1,:, p,3:7]  # N x 4
            orient_inv = quatconj(orient)               # L x N x 4
            orient_inv = orient_inv.view(-1, n_inst, 1, 4)  # L x N x 1 x 4 or 1 x N x 1 x 4
            origin = origin.unsqueeze(-2)               # (L x) N x 1 x 3

        ob_link_pos = link_pos - origin                                     # L x N x n_links x 3 
        ob_link_pos = rotatepoint(orient_inv, ob_link_pos)
        ob_link_orient = quatmultiply(orient_inv, link_orient)  # L x N x n_links x 4

        if include_velocity:
            if k[0] == -1:
                link_lin_vel, link_ang_vel = link_tensor[...,7:10], link_tensor[...,10:13]
            else:
                link_lin_vel, link_ang_vel = link_tensor[:,:,k,7:10], link_tensor[:,:,k,10:13]
            ob_link_lin_vel = rotatepoint(orient_inv, link_lin_vel)         # L x N x n_links x 3
            ob_link_ang_vel = rotatepoint(orient_inv, link_ang_vel)         # L x N x n_links x 3
            ob = torch.cat((ob_link_pos, ob_link_orient,
                ob_link_lin_vel, ob_link_ang_vel), -1)                      # L x N x n_links x 13
        else:
            ob = torch.cat((ob_link_pos, ob_link_orient), -1)               # L x N x n_links x 7
        ob = ob.view(n_hist, n_inst, -1)                                    # L x N x (n_links x 7 or 13)
        obs.append(ob)
    ob = torch.cat((obs), -1)

    ob1 = ob.permute(1, 0, 2)                                           # N x L x (n_links x 7 or 13)
    if seq_len is None: return ob1

    ob2 = torch.zeros_like(ob1)
    arange = torch.arange(n_hist, dtype=seq_len.dtype, device=seq_len.device).unsqueeze_(0)
    seq_len_ = seq_len.unsqueeze(1)
    mask1 = arange > (n_hist-1) - seq_len_
    mask2 = arange < seq_len_
    ob2[mask2] = ob1[mask1]
    return ob2


def ICCGANTarget_wrapper(base_class):
    class ICCGANTarget(base_class):
        GOAL_DIM = 4
        GOAL_TENSOR_DIM = 2
        ENABLE_GOAL_TIMER = True

        GOAL_RADIUS = 0.5
        SP_LOWER_BOUND = 1.2
        SP_UPPER_BOUND = 1.5
        GOAL_TIMER_RANGE = 90, 150
        GOAL_SP_MEAN = 1.2
        GOAL_SP_STD = 0
        GOAL_SP_MIN = 1.2
        GOAL_SP_MAX = 1.2
        GOAL_TIMER_THRESHOLD = 0

        SHARP_TURN_RATE = 1

        def __init__(self, *args, **kwargs):
            self.goal_radius = kwargs.get("goal_radius", self.GOAL_RADIUS)
            self.sharp_turn_rate = kwargs.get("sharp_turn_rate", self.SHARP_TURN_RATE)
            self.sp_lower_bound = kwargs.get("sp_lower_bound", self.SP_LOWER_BOUND)
            self.sp_upper_bound = kwargs.get("sp_upper_bound", self.SP_UPPER_BOUND)
            self.goal_timer_range = kwargs.get("goal_timer_range", self.GOAL_TIMER_RANGE)
            self.goal_timer_threshold = kwargs.get("goal_timer_threshold", self.GOAL_TIMER_THRESHOLD)
            self.goal_sp_mean = kwargs.get("goal_sp_mean", self.GOAL_SP_MEAN)
            self.goal_sp_std = kwargs.get("goal_sp_std", self.GOAL_SP_STD)
            self.goal_sp_min = kwargs.get("goal_sp_min", self.GOAL_SP_MIN)
            self.goal_sp_max = kwargs.get("goal_sp_max", self.GOAL_SP_MAX)
            super().__init__(*args, **kwargs)

        def update_viewer(self):
            super().update_viewer()
            self.gym.clear_lines(self.viewer)
            n_lines = 10
            tar_x = self.goal_tensor[:, 0].cpu().numpy()

            p = self.root_pos.cpu().numpy()
            zero = np.zeros_like(tar_x)+0.05
            tar_y = self.goal_tensor[:, 1].cpu().numpy()
            lines = np.stack([
                np.stack((p[:,0], p[:,1], zero+0.01*i, tar_x, tar_y, zero), -1)
            for i in range(n_lines)], -2)
            for e, l in zip(self.envs, lines):
                self.gym.add_lines(self.viewer, e, n_lines, l, [[1., 0., 0.] for _ in range(n_lines)])
            n_lines = 10
            target_pos = self.goal_tensor.cpu().numpy()
            lines = np.stack([
                np.stack((
                    target_pos[:, 0], target_pos[:, 1], zero,
                    target_pos[:, 0]+self.goal_radius*np.cos(2*np.pi/n_lines*i), 
                    target_pos[:, 1]+self.goal_radius*np.sin(2*np.pi/n_lines*i),
                    zero
                ), -1)
            for i in range(n_lines)], -2)
            for e in self.envs:
                self.gym.add_lines(self.viewer, e, n_lines, lines, [[0., 0., 1.] for _ in range(n_lines)])
        
        @staticmethod
        def _observe_goal(self, env_ids):
            if env_ids is None:
                if self.goal_timer_threshold:
                    timer = (self.goal_timer-self.goal_timer_threshold).clip_(min=1)
                else:
                    timer = self.goal_timer
                g, self.v_target = observe_iccgan_target(
                    self.root_tensor[:, 0], self.goal_tensor, timer*self.step_time,
                    sp_lower_bound=self.sp_lower_bound, sp_upper_bound=self.sp_upper_bound,
                    goal_radius=self.goal_radius
                )
            else:
                if self.goal_timer_threshold:
                    timer = (self.goal_timer[env_ids]-self.goal_timer_threshold).clip_(min=1)
                else:
                    timer = self.goal_timer[env_ids]
                g, self.v_target[env_ids] = observe_iccgan_target(
                    self.root_tensor[env_ids, 0], self.goal_tensor[env_ids], timer*self.step_time,
                    sp_lower_bound=self.sp_lower_bound, sp_upper_bound=self.sp_upper_bound,
                    goal_radius=self.goal_radius
                )
            return g

        def _observe(self, env_ids):
            ob = super()._observe(env_ids)
            g = ICCGANTarget._observe_goal(self, env_ids)
            return torch.cat((ob, g), -1)

        def reset_goal(self, env_ids, goal_tensor=None, goal_timer=None):
            if goal_tensor is None: goal_tensor = self.goal_tensor
            if goal_timer is None: goal_timer = self.goal_timer
            return self.__class__._reset_goal(self, env_ids, goal_tensor, goal_timer)
        
        @staticmethod
        def _reset_goal(self, env_ids, goal_tensor, goal_timer):
            n_envs = len(env_ids)
            all_envs = n_envs == len(self.envs)
            root_orient = self.root_orient if all_envs else self.root_orient[env_ids]

            small_turn = torch.rand(n_envs, device=self.device) > self.sharp_turn_rate
            large_angle = torch.rand(n_envs, dtype=torch.float32, device=self.device).mul_(2*np.pi)
            small_angle = torch.rand(n_envs, dtype=torch.float32, device=self.device).sub_(0.5).mul_(2*(np.pi/3))

            heading = heading_zup(root_orient)
            small_angle += heading
            theta = torch.where(small_turn, small_angle, large_angle)

            timer = torch.randint(self.goal_timer_range[0], self.goal_timer_range[1], (n_envs,), dtype=self.goal_timer.dtype, device=self.device)
            if self.goal_sp_min == self.goal_sp_max:
                vel = self.goal_sp_min
            elif self.goal_sp_std == 0:
                vel = self.goal_sp_mean
            else:
                vel = torch.nn.init.trunc_normal_(torch.empty(n_envs, dtype=torch.float32, device=self.device), mean=self.goal_sp_mean, std=self.goal_sp_std, a=self.goal_sp_min, b=self.goal_sp_max)
                vel.clip_(min=0)
                
            dist = vel*timer*self.step_time
            dx = dist*torch.cos(theta)
            dy = dist*torch.sin(theta)

            if self.goal_timer_threshold:
                timer += self.goal_timer_threshold
            
            if all_envs:
                goal_timer.copy_(timer)
                goal_tensor[:,0] = self.root_pos[:,0] + dx
                goal_tensor[:,1] = self.root_pos[:,1] + dy
            else:
                goal_timer.index_copy_(0, env_ids, timer)
                goal_tensor[env_ids,0] = self.root_pos[env_ids,0] + dx
                goal_tensor[env_ids,1] = self.root_pos[env_ids,1] + dy
            

        def _reward_target(self):
            p = self.root_pos[:, :2]
            p_ = self.state_hist[-1, :, :2]

            p_tar = self.goal_tensor[:,:2]
            v_tar = self.v_target
            sp_tar = torch.linalg.norm(v_tar, ord=2, dim=-1)

            v = (p - p_).mul_(self.fps)
            r = (v - v_tar).square_().sum(1).mul_(-2/(sp_tar*sp_tar).clip_(min=1)).exp_()

            dp = p_tar - p
            dist = torch.linalg.norm(dp, ord=2, dim=-1)
            dist[dist<1e-5] = 0
            near = dist < self.goal_radius
            r[near] = 1

            self.info["near"] = near
            return r.unsqueeze_(-1)

    return ICCGANTarget

@torch.jit.script
def observe_iccgan_target(root_tensor: torch.Tensor, 
    target_tensor: torch.Tensor, timer: torch.Tensor,
    sp_lower_bound:float, sp_upper_bound: float,
    goal_radius:float
):
    root_pos = root_tensor[:, :2]
    root_orient = root_tensor[:, 3:7]

    dp = target_tensor[:,:2] - root_pos

    dist = dp.square().sum(-1).sqrt_()
    sp = dist.div(timer).unsqueeze_(-1)

    dist[dist<1e-5] = 0
    dist.unsqueeze_(-1)
    dp.div_(dist).nan_to_num_(0.,0.,0.)
    
    sp.clip_(min=sp_lower_bound, max=sp_upper_bound)
    sp[dist<goal_radius] *= 0
    v_target = dp*sp

    dist.div_(2.5).clip_(max=2).add_(-1)
    sp.div_(sp_upper_bound*0.5).add_(-1)
    
    x, y = dp[:,0], dp[:,1]
    heading_inv = -heading_zup(root_orient)
    c = torch.cos(heading_inv)
    s = torch.sin(heading_inv)
    x, y = c*x-s*y, s*x+c*y

    return torch.cat((x.unsqueeze_(-1), y.unsqueeze_(-1), sp, dist), -1), v_target


_ICCGANTarget = ICCGANTarget_wrapper(ICCGANHumanoid)
  
class ICCGANTargetDefense(_ICCGANTarget):
    GOAL_DIM = 5
    GOAL_TENSOR_DIM = 3
    GOAL_REWARD_WEIGHT = 0.5

    def create_tensors(self):
        super().create_tensors()
        n_moves = 3
        self.sampling_cnts = np.ones((n_moves,), dtype=np.int64)
        self.arange_n_envs = torch.arange(len(self.envs), device=self.device)

        rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], self.gym.get_actor_handle(self.envs[0], 0))
        self.palms = [rigid_body["RH:palm"], rigid_body["LH:palm"]]
        self.wrists = [rigid_body["right_hand"], rigid_body["left_hand"]]
        self.elbows = [rigid_body["right_lower_arm"], rigid_body["left_lower_arm"]]
        self.shoulders = [rigid_body["right_upper_arm"], rigid_body["left_upper_arm"]]

    def reset_goal(self, env_ids, goal_tensor=None, goal_timer=None):
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer is None: goal_timer = self.goal_timer

        n_envs = len(env_ids)
        all_envs = n_envs == len(self.envs)
        root_orient = self.root_orient if all_envs else self.root_orient[env_ids]

        small_turn = torch.rand(n_envs, device=self.device) > self.sharp_turn_rate
        large_angle = torch.rand(n_envs, dtype=torch.float32, device=self.device).mul_(2*np.pi)
        small_angle = torch.rand(n_envs, dtype=torch.float32, device=self.device).sub_(0.5).mul_(2*(np.pi/3))

        heading = heading_zup(root_orient)
        small_angle += heading
        theta = torch.where(small_turn, small_angle, large_angle)

        timer = torch.randint(self.goal_timer_range[0], self.goal_timer_range[1], (n_envs,), dtype=self.goal_timer.dtype, device=self.device)
        if self.goal_sp_min == self.goal_sp_max:
            vel = self.goal_sp_min
        elif self.goal_sp_std == 0:
            vel = self.goal_sp_mean
        else:
            vel = torch.nn.init.trunc_normal_(torch.empty(n_envs, dtype=torch.float32, device=self.device), mean=self.goal_sp_mean, std=self.goal_sp_std, a=self.goal_sp_min, b=self.goal_sp_max)
            vel.clip_(min=0)
            
        dist = vel*timer*self.step_time
        dx = dist*torch.cos(theta)
        dy = dist*torch.sin(theta)

        if self.goal_timer_threshold:
            timer += self.goal_timer_threshold
        
        if self.training:
            if self.simulation_step < 40000:
                n_envs_inplace = len(self.envs)
            else:
                n_envs_inplace = len(self.envs) * max(0, 1-(self.simulation_step - 40000)/40000)
        n_envs_inplace = len(self.envs)
        in_place_envs = self.arange_n_envs < n_envs_inplace

        if all_envs:
            goal_timer.copy_(timer)
            m = (self.lifetime == 0).logical_and_(in_place_envs)
            goal_tensor[m, :2] = self.root_pos[m, :2]
            m2 = ~in_place_envs
            goal_tensor[m2, 0] = self.root_pos[m2, 0] + dx[m2]
            goal_tensor[m2, 1] = self.root_pos[m2, 1] + dy[m2]
        else:
            goal_timer.index_copy_(0, env_ids, timer)
            m = env_ids[(self.lifetime[env_ids] == 0).logical_and_(env_ids < n_envs_inplace)]
            goal_tensor[m, :2] = self.root_pos[m, :2]
            m20 = env_ids >= n_envs_inplace
            m2 = env_ids[m20]
            goal_tensor[m2, 0] = self.root_pos[m2, 0] + dx[m20]
            goal_tensor[m2, 1] = self.root_pos[m2, 1] + dy[m20]

        if self.training:
            if np.all(self.sampling_cnts > 10000):
                self.sampling_cnts -= 10000
            weight = np.sum(self.sampling_cnts)/self.sampling_cnts
            weight /= np.sum(weight)
            self.goal_tensor[env_ids, -1] = torch.tensor(np.random.choice(np.arange(len(weight)), size=(len(env_ids), ), replace=True, p=weight), dtype=torch.float, device=self.device)
        else:
            self.goal_tensor[env_ids, -1] = torch.randint(len(self.sampling_cnts), (len(env_ids),), device=self.device, dtype=torch.float32)

    def _observe(self, env_ids):
        if env_ids is None:
            goal_action = self.goal_tensor[:, -1:]
        else:
            goal_action = self.goal_tensor[env_ids, -1:]
        g = (goal_action/2).add_(-1)
        return torch.cat((super()._observe(env_ids), g), -1)

    def reward(self):
        p = self.root_pos[:, :2]
        p_ = self.state_hist[-1, :, :2]

        p_tar = self.goal_tensor[:,:2]
        v_tar = self.v_target
        sp_tar = torch.linalg.norm(v_tar, ord=2, dim=-1)

        v = (p - p_).mul_(self.fps)
        r = (v - v_tar).square_().sum(1).mul_(-2/(sp_tar*sp_tar).clip_(min=1)).exp_()

        dp = p_tar - p
        dist = torch.linalg.norm(dp, ord=2, dim=-1)
        dist[dist<1e-5] = 0
        near = dist < self.goal_radius
        r[near] = v[near].square().sum(-1).neg_().exp_().mul_(0.2).add_(1.)

        self.info["near"] = near


        arm_dir = self.link_pos[:, self.elbows] - self.link_pos[:, self.shoulders]
        angle0 = torch.atan2(arm_dir[..., 2], arm_dir[..., :2].norm(p=2, dim=-1))
        angle0.div_(np.pi)
        arm_dir = self.link_pos[:, self.wrists] - self.link_pos[:, self.elbows]
        angle1 = torch.atan2(arm_dir[..., 2], arm_dir[..., :2].norm(p=2, dim=-1))
        angle1.div_(np.pi)
        d_hand = (self.link_pos[:, self.palms[0]]-self.link_pos[:, self.palms[1]]).square_().sum(-1).sqrt_()

        r_block = ((angle0+0.212)/0.376).clip_(max=1).max(-1).values*0.75+0.25*(angle1/0.166).clip_(max=1).max(-1).values
        r_screen = torch.minimum(
                ((0.5-angle1)/0.8).clip_(max=1).min(-1).values,
                ((0.4-angle0)/0.8).clip_(max=1).min(-1).values,
            )*0.75+0.25*((0.5-d_hand)/0.3).clip_(max=1)
        r_defense = ((0.5-angle0.abs())/0.334).clip_(max=1).min(-1).values
        r_relax = ((0.4-angle0)/0.8).clip_(max=1).min(-1).values*0.75+0.25*((d_hand-0.075)/0.125).clip_(max=1) 

        action_goal = self.goal_tensor[:,-1]
        c_defense = action_goal == 0
        c_screen = action_goal == 1
        c_block = action_goal == 2

        rew_def = torch.where(c_defense,
            r_defense, torch.where(c_screen,
            r_screen, torch.where(c_block,
            r_block, r_relax
        )))

        r[near] += rew_def[near]*0.8

        if self.training and self.simulation_step > 0 :
            cnts_block = torch.sum(c_block.logical_and_(near)).item()
            cnts_screen = torch.sum(c_screen.logical_and_(near)).item()
            cnts_defense = torch.sum(c_defense.logical_and_(near)).item()
            self.sampling_cnts[0] += cnts_defense
            self.sampling_cnts[1] += cnts_screen
            self.sampling_cnts[2] += cnts_block
            if len(self.sampling_cnts) > 3:
                cnts_relax = torch.sum(near).item() - cnts_block - cnts_screen - cnts_defense
                self.sampling_cnts[3] += cnts_relax
        return r.unsqueeze_(-1)
    

class ICCGANBall(ICCGANHumanoid):

    BALL_RADIUS = 0.1152 # meters
    BALL_MASS = 0.56699  # kg
    BASKET_RADIUS = 0.225 # meters, inner radius
    BASKET_HEIGHT = 3.05
    BASKET_POS = 12.4 # olympic dimension

    def __init__(self, *args, **kwargs):
        self.goal_horizon = kwargs.get("goal_horizon", 9)

        if "frameskip" not in kwargs: kwargs["frameskip"] = 1
        if "substeps" not in kwargs: kwargs["substeps"] = 4
        super().__init__(*args, **kwargs)
    
    def register_asset(self):
        assets = super().register_asset()
        court_asset_options = gymapi.AssetOptions()
        court_asset_options.fix_base_link = True
        court_asset_options.disable_gravity = True
        court_asset_options.vhacd_enabled = True
        court_asset_options.vhacd_params.max_convex_hulls = 32
        court_asset_options.vhacd_params.max_num_vertices_per_ch = 64
        court_asset_options.vhacd_params.resolution = 300000
        court_asset = self.gym.load_asset(self.sim, os.path.abspath(os.path.dirname("assets/court.xml")), "court.xml", court_asset_options)
        assets["court"] = court_asset

        ball_properties = dict(
            compliance=0.0,
            friction=1.0,
            restitution=1.,
            rolling_friction=1.0,
            torsion_friction=1.0
        )
        ball_asset_options = gymapi.AssetOptions()
        ball_asset_options.density = self.BALL_MASS / (4*np.pi*(self.BALL_RADIUS**3)/3)
        ball_asset = self.gym.create_sphere(self.sim, self.BALL_RADIUS, ball_asset_options)
        props = self.gym.get_asset_rigid_shape_properties(ball_asset)
        for k, v in ball_properties.items(): setattr(props[0], k, v)
        self.gym.set_asset_rigid_shape_properties(ball_asset, props)
        assets["ball"] = ball_asset
        return assets

    def add_actor(self, env, env_id, assets):
        ball_pose = gymapi.Transform()
        ball_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        ball_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        ball_actor = self.gym.create_actor(env, assets["ball"], ball_pose, "ball", env_id, -1, 0)

        court_pose = gymapi.Transform()
        court_pose.p = gymapi.Vec3(0, 0, 0)
        court_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        court_actor = self.gym.create_actor(env, assets["court"], court_pose, "court", env_id, -1, 0)

        for aid in range(self.gym.get_actor_count(env)):
            actor = self.gym.get_actor_handle(env, aid)
            rb_shape = self.gym.get_actor_rigid_body_shape_indices(env, actor)
            rb_shape_props = self.gym.get_actor_rigid_shape_properties(env, actor)
            for n in self.gym.get_actor_rigid_body_names(env, actor):
                if n in ["right_hand", "left_hand"] or "LH:" in n or "RH:" in n:
                    link = self.gym.find_actor_rigid_body_index(env, actor, n, gymapi.DOMAIN_ACTOR)
                    if link == -1 or rb_shape[link].count < 1: continue
                    for i in range(rb_shape[link].count):
                        rb_shape_props[rb_shape[link].start+i].friction = 1
                        rb_shape_props[rb_shape[link].start+i].rolling_friction = 1
                        rb_shape_props[rb_shape[link].start+i].torsion_friction = 1
            self.gym.set_actor_rigid_shape_properties(env, actor, rb_shape_props)
        
        return ball_actor, court_actor
    
    def create_tensors(self):
        super().create_tensors()
        self.natural_dv = -9.81/self.fps    # velocity change between frames without external force

        self.ball_root_tensor = self.root_tensor[:, -2]
        self.ball_link_tensor = self.link_tensor[:, -2]
        self.ball_actor_ids = self.actor_ids[:, -2].flatten().contiguous()

        self.ball_init_pose = torch.zeros((len(self.envs), 13), device=self.device)
        self.ball_init_pose[..., 6] = 1
        self.court_init_pose = torch.zeros((len(self.envs), 1, 13), device=self.device)
        self.court_init_pose[..., 6] = 1

    def build_motion_lib(self, motion_file):
        return ReferenceMotion(motion_file=motion_file, character_model=self.character_model+["assets/ball.xml"],
            key_links=None, device=self.device)


class ICCGANDribble(ICCGANBall):

    GOAL_DIM = 9+1

    def reset(self):
        super().reset()
        self.info["success_rate"] = 0.
    
    def create_tensors(self):
        super().create_tensors()
        n_envs = len(self.envs)
        self.launched = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.can_dribble = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.ball_vel = torch.zeros((n_envs,3), dtype=torch.float32, device=self.device)
        self.contact_ground_counter = torch.zeros((n_envs,), dtype=torch.int32, device=self.device)
        self.arange_tensor = torch.arange(n_envs, device=self.device)

        rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], self.gym.get_actor_handle(self.envs[0], 0))
        self.palms = [rigid_body["RH:palm"], rigid_body["LH:palm"]]
        self.elbows = [rigid_body["right_lower_arm"], rigid_body["left_lower_arm"]]
        self.shoulders = [rigid_body["right_upper_arm"], rigid_body["left_upper_arm"]]
        self.palms_rotation_axis = torch.tensor([[[0, 1, 0], [0, -1, 0]]], dtype=torch.float, device=self.device)
        self.left_finger_tips_with_palm = sorted([i for n, i in rigid_body.items() if "LH:" in n and ("tip" in n or "palm" in n)])
        self.right_finger_tips_with_palm = sorted([i for n, i in rigid_body.items() if "RH:" in n and ("tip" in n or "palm" in n)])
        self.left_finger_tips = sorted([i for n, i in rigid_body.items() if "LH:" in n and ("tip" in n)])
        self.right_finger_tips = sorted([i for n, i in rigid_body.items() if "RH:" in n and ("tip" in n)])
        self.finger_tips = self.right_finger_tips+self.left_finger_tips
        assert all([_ > 0 for _ in self.palms])
        assert all([_ > 0 for _ in self.left_finger_tips_with_palm])
        assert all([_ > 0 for _ in self.right_finger_tips_with_palm])

    @staticmethod
    def reset_ball(self, env_ids, ref_link_tensor):
        n_envs = len(env_ids)
        y = torch.rand((n_envs,1), dtype=torch.float32, device=self.device)
        z = torch.rand_like(y)
        y.mul_(0.6).add_(-0.3) # +/- 0.3
        z.mul_(0.3).add_(0.6)  # 0.6-0.9
        x = torch.rand_like(y)
        x.mul_(0.3).add_(0.5)  # 0.5-0.8

        heading = heading_zup(ref_link_tensor[:, :1, 3:7])
        c, s = torch.cos(heading), torch.sin(heading)

        link_pos = ref_link_tensor[..., :2] - ref_link_tensor[:, :1, :2]
        offset = (c*link_pos[..., 0] + s*link_pos[..., 1]).max(-1).values.clip_(min=0)
        x = x + offset.unsqueeze_(-1)
        x, y = c*x-s*y+ref_link_tensor[:, 0, :1], s*x+c*y+ref_link_tensor[:, 0, 1:2]
        self.ball_init_pose[env_ids,0] = x
        self.ball_init_pose[env_ids,1] = y
        self.ball_init_pose[env_ids,2] = z
        self.ball_init_pose[env_ids,9] = -self.natural_dv

        self.ball_vel.index_copy_(0, env_ids, self.ball_init_pose[env_ids,7:10])
        self.launched.index_fill_(0, env_ids, False)
        self.contact_ground_counter.index_fill_(0, env_ids, 0)
        self.can_dribble.index_fill_(0, env_ids, True)

    @staticmethod
    def _init_state(self, env_ids, ref_motion=None):
        if ref_motion is None: ref_motion = self.ref_motion
        motion_ids, motion_times = ref_motion.sample(len(env_ids))
        ref_link_tensor, ref_joint_tensor = ref_motion.state(motion_ids, motion_times)
        ICCGANDribble.reset_ball(self, env_ids, ref_link_tensor[:, :-1])
        ref_link_tensor[:, -1] = self.ball_init_pose[env_ids]
        return ref_link_tensor, ref_joint_tensor

    def init_state(self, env_ids):
        ref_link_tensor, ref_joint_tensor = self._init_state(self, env_ids)
        ref_link_tensor = torch.cat((ref_link_tensor, self.court_init_pose[env_ids]), 1)
        return ref_link_tensor, ref_joint_tensor
    
    @staticmethod
    def _observe_goal(self, env_ids):
        if env_ids is None:
            can_dribble = self.can_dribble
            g = observe_dribble(self.root_tensor[:, 0], self.ball_root_tensor)
        else:
            can_dribble = self.can_dribble[env_ids]
            g = observe_dribble(self.root_tensor[env_ids, 0], self.ball_root_tensor[env_ids])
        return g, can_dribble.unsqueeze(-1)

    def _observe(self, env_ids):
        ob = super()._observe(env_ids)
        g = ICCGANDribble._observe_goal(self, env_ids)
        return torch.cat((ob, *g), -1)

    def reward(self):
        return self.__class__._reward_dribble(self)

    @staticmethod
    def _reward_dribble(self):
        ball_pos = self.ball_root_tensor[:, :3]
        ball_vel = self.ball_root_tensor[:, 7:10]

        vv = ball_vel[:, 2]
        dv = vv - self.ball_vel[:, 2]
        free_falling = (dv - self.natural_dv).abs_() < 1e-4
        launch = torch.logical_and(~self.launched, ~free_falling)
        self.launched[launch] = True
        touch = dv < self.natural_dv - 1e-2
        dribble = (vv < 0).logical_and_(touch)

        down_to_up = torch.logical_and(self.ball_vel[:, 2] <= 0, ball_vel[:, 2] > 0)
        up_to_down = torch.logical_and(self.ball_vel[:, 2] >= 0, ball_vel[:, 2] < 0)
        dribble_miss = torch.logical_and(self.can_dribble, down_to_up)

        palm_pos = self.link_pos[:, self.palms, :3]
        face_dir = rotatepoint(self.link_orient[:, self.palms], self.palms_rotation_axis)
        dist30 = (palm_pos + face_dir*self.BALL_RADIUS - ball_pos.unsqueeze(1)).square_().sum(-1)
        e = ((ball_pos.unsqueeze(1)-palm_pos)*face_dir).sum(-1).clip_(min=0)
        dist20 = (palm_pos + face_dir*e.unsqueeze(2) - ball_pos.unsqueeze(1)).square_().sum(-1)

        dist3 = dist30.min(-1)
        nearest_hand = dist3.indices

        dribble_by_right = nearest_hand == 0
        dist2 = torch.where(dribble_by_right, dist20[:, 0], dist20[:, 1])

        dist3 = dist3.values

        r_dist = torch.where(torch.logical_or(self.can_dribble, touch),
            dist3.sqrt().mul_(-2),
            dist2.mul(-5)
        )
        r_dist.exp_()

        # vt+0.5gt^2 = h-r
        # t = (sqrt(v^2 - 2g(h-r))-v)/g
        #
        # a(v+gt) T - 0.5gT^2 = H-r
        # T = a(v+gt)/g = a sqrt(v^2 + 2g(h-r)) / g
        # 0.5gT^2 = H-r
        # 0.5 a^2 (v^2 + 2g(h-r)) / g = H-r
        # v = sqrt(2g ((H-r) / a^2 - (h-r)))
        h = self.link_pos[:, 0, 2]+self.BALL_RADIUS*3
        v_target = torch.where(vv>0,
            (h-ball_pos[:,2]).mul_(2*9.81).sqrt_(),
            (h/(0.875*0.875)).sub_(ball_pos[:,2]).mul_(2*9.81).sqrt_(),
        )
        sp = vv.abs().div_(v_target).nan_to_num_(nan=1.0, posinf=1.0, neginf=1.0)*self.launched
        r_sp = sp.clip(max=1.)*self.launched

        key_fingers_left = self.left_finger_tips_with_palm
        key_fingers_right = self.right_finger_tips_with_palm
        left_ball_dist = torch.linalg.norm(self.link_pos[:, key_fingers_left] - ball_pos.unsqueeze(1), ord=2, axis=-1)
        right_ball_dist = torch.linalg.norm(self.link_pos[:, key_fingers_right] - ball_pos.unsqueeze(1), ord=2, axis=-1)
        bh_dist = torch.where(dribble_by_right.unsqueeze(-1), right_ball_dist, left_ball_dist)
        r_dribble = 0.2+(bh_dist-self.BALL_RADIUS).clip_(min=0).sum(-1).mul_(-10).exp_()*0.8
        r_dribble *= dribble

        rew_dribble = 0.6*r_dist + 0.4*r_sp + 0.5*r_dribble
        rew_dribble -= 1.*(~self.launched)

        h = ball_pos[..., 2]
        push_up  = (h > 0.3).logical_and_(down_to_up)
        too_low  = (h < 0.3).logical_and_(up_to_down)
        too_high =  h > self.link_pos[:, 0, 2]+0.5
        contact_ground = h < self.BALL_RADIUS*2+0.01
        self.contact_ground_counter *= contact_ground
        self.contact_ground_counter += contact_ground
        contact_ground = self.contact_ground_counter > 3
        too_far = (ball_pos[..., :2] - self.link_pos[:, 0, :2]).square_().sum(-1) > 2.25
        reset = push_up.logical_or_(too_low).logical_or_(too_high).logical_or_(too_far).logical_or_(contact_ground)
        contact = torch.any((self.contact_force_tensor[:, (0, 1, 2, 3, 4, 27, 28)].abs() > 1).view(reset.size(0), -1), -1)
        reset.logical_or_(contact)

        if self.simulation_step > 0:
            # ignore the first dribble to launch
            dribble_succ = (~launch).logical_and_(self.launched).logical_and_(self.can_dribble).logical_and_(dribble)
            dribble_miss += reset
            n = torch.sum(torch.logical_or(dribble_succ, dribble_miss)).item()
            if n > 0:
                succ_rate = torch.sum(torch.logical_and(dribble_succ, ~dribble_miss)).item()*1.0/n
                self.info["success_rate"] *= 0.9
                self.info["success_rate"] += succ_rate*0.1

        self.info["reset"] = reset
        env_ids = self.arange_tensor[:ball_pos.size(0)][reset]
        rew_dribble[env_ids] = -1

        # should not dribble if the ball is dribbled until it bounds on the ground
        self.can_dribble += down_to_up
        self.can_dribble *= ~dribble

        if env_ids.numel() > 0:
            ICCGANDribble.reset_ball(self, env_ids, self.link_tensor[reset, :-2])
            reset.logical_or_(~self.launched)
        else:
            reset = ~self.launched
        env_ids = self.ball_actor_ids[reset]
        if env_ids.numel() > 0:
            self.ball_root_tensor[reset] = self.ball_init_pose[reset]
            self.ball_link_tensor[reset] = self.ball_init_pose[reset]
            self.root_updated_actors.append(env_ids)
        # self.ball_pos.copy_(self.ball_root_tensor[:, :3])
        self.ball_vel.copy_(self.ball_root_tensor[:, 7:10])

        return rew_dribble.unsqueeze_(-1)
    
@torch.jit.script
def observe_dribble(root_tensor: torch.Tensor, ball_tensor: torch.Tensor):    
    orient = root_tensor[:, 3:7]
    origin = root_tensor[:, :3].clone()
    origin[..., 2] = 0                                          # N x 3
    heading = heading_zup(orient)
    up_dir = torch.zeros_like(origin)
    up_dir[..., 2] = 1
    orient_inv = axang2quat(up_dir, -heading)                   # N x 4

    ball_pos = rotatepoint(orient_inv, ball_tensor[..., :3]-origin)
    ball_vel = rotatepoint(orient_inv.unsqueeze(1), ball_tensor[..., 7:13].view(-1, 2, 3))
    return torch.cat((ball_pos, ball_vel.view(-1, 6)), -1)


class ICCGANDribbleTarget(ICCGANTarget_wrapper(ICCGANDribble)):
    GOAL_DIM = ICCGANDribble.GOAL_DIM + 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nav_reward_weight = self.reward_weights[0, -1].item()
        dribble_reward_weight = self.reward_weights[0, -2].item()
        self.task_reward_weight = self.nav_reward_weight + dribble_reward_weight
        self.nav_reward_weight_inc = self.task_reward_weight * 0.5 - self.nav_reward_weight
        
    @staticmethod
    def _reward_nav(self):
        p = self.root_pos[:, :2]
        p_ = self.state_hist[-1, :, :2]

        p_tar = self.goal_tensor[:,:2]
        v_tar = self.v_target
        sp_tar = torch.linalg.norm(v_tar, ord=2, dim=-1)

        v = (p - p_).mul_(self.fps)
        r = (v - v_tar).square_().sum(1).mul_(-2/(sp_tar*sp_tar).clip_(min=1)).exp_()

        dp = p_tar - p
        dist = torch.linalg.norm(dp, ord=2, dim=-1)
        dist[dist<1e-5] = 0
        near = dist < self.goal_radius
        r += 0.5*near*dist.mul(-1/self.goal_radius).exp_()

        self.info["near"] = near
        return r.unsqueeze_(-1)
    
    def reward(self):
        rew_dribble = self.__class__._reward_dribble(self)
        rew_nav = self.__class__._reward_nav(self)

        if self.simulation_step:
            # navigation reward
            self.reward_weights[:, -1].clip_(min=self.nav_reward_weight+self.nav_reward_weight_inc*min(np.exp(10*(self.info["success_rate"]-1)), 1))
            # dribble reward
            self.reward_weights[:, -2] = self.task_reward_weight-self.reward_weights[:, -1]
        return torch.cat((rew_dribble, rew_nav), -1)


class ICCGANShoot(ICCGANBall):
    GOAL_DIM = 9+3
    CAMERA_POS = 0, -8, 3.0
        
    def reset(self):
        super().reset()
        self.info["shot_percent"] = []

    def create_tensors(self):
        super().create_tensors()
        self.episode_length = 60

        n_envs = len(self.envs)
        self.caught = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.released = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.fallen = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.ball_pos = torch.zeros((n_envs,3), dtype=torch.float32, device=self.device)
        self.ball_vel = torch.zeros((n_envs,3), dtype=torch.float32, device=self.device)
        self.ball_height_before_release = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.ball_height_max = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.ball_shoot_reward = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.foot_contacted = torch.zeros((n_envs,2), dtype=torch.bool, device=self.device)
        self.pivot_foot = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.pivot_foot_set_at = torch.zeros((n_envs,), dtype=torch.int64, device=self.device)
        self.foot_contact_pos = torch.zeros((n_envs,2,4,3), dtype=torch.float32, device=self.device)
        self.foot_contact_vert = torch.zeros((n_envs,2,4,1), dtype=torch.bool, device=self.device)
        self.arange_2nenvs = torch.arange(n_envs*2, dtype=torch.int64, device=self.device)
        self.zeros1 = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.ref_dir_z = torch.zeros((3,), dtype=torch.float32, device=self.device)
        self.ref_dir_z[2] = 1

        rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], self.gym.get_actor_handle(self.envs[0], 0))

        self.palms = [rigid_body["RH:palm"], rigid_body["LH:palm"]]
        self.palms_rotation_axis = torch.tensor([
            [[0, 1, 0], [0, -1, 0]]
        ], dtype=torch.float, device=self.device)

        self.left_finger_tips_with_palm = sorted([i for n, i in rigid_body.items() if "LH:" in n and ("tip" in n or "palm" in n)])
        self.right_finger_tips_with_palm = sorted([i for n, i in rigid_body.items() if "RH:" in n and ("tip" in n or "palm" in n)])
        
        self.foot_links = [rigid_body["right_foot"], rigid_body["left_foot"]]
        self.foot_vertices = torch.tensor([[[
            [0.045+0.0885,  0.045, -0.0225-0.0275],
            [0.045+0.0885, -0.045, -0.0225-0.0275],
            [0.045-0.0885,  0.045, -0.0225-0.0275],
            [0.045-0.0885, -0.045, -0.0225-0.0275]
        ]]], dtype=torch.float32, device=self.device)
        
        self.wrist_links = [rigid_body["right_hand"], rigid_body["left_hand"]]

    def _init_state(self, env_ids, zero_vel=True, max_time=0):
        n_envs = len(env_ids)
        motion_ids, motion_times = self.ref_motion.sample(n_envs, max_time=max_time)
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)

        # All motion heading direction should be correctized during data processing
        # such that the shoot direction is along positive x-axis direction
        ref_link_tensor[:, :, 0] -= ref_link_tensor[:, :1, 0].clone()

        random_dist = True
        random_dir = True
        noise_on_facing_dir = True

        if random_dist:
            if random_dir:
                # uniformly sample on a 2d ring range with inner radius 2.5 and outer radius 7.5
                # d = (u * (R**2 - r**2) + r**2)**(1/2)
                d = torch.rand((n_envs,1), dtype=torch.float32, device=self.device).mul_(50).add_(6.25).sqrt_()
                d.neg_()
            else:
                d = torch.rand((n_envs,1), dtype=torch.float32, device=self.device).mul_(-5).add_(-2.5)
        else:
            d = -6.75
        if noise_on_facing_dir:
            theta_noise = torch.rand((n_envs,1), dtype=torch.float32, device=self.device).mul_(1.0472).add_(-0.5236)
        if random_dir:
            theta = torch.rand((n_envs,1), dtype=torch.float32, device=self.device).mul_(3.56).add_(-1.78)
            y, x = torch.sin(theta)*d, self.BASKET_POS + torch.cos(theta)*d
            if noise_on_facing_dir:
                theta += theta_noise
        else:
            x = self.BASKET_POS + d
            y = 0
            if noise_on_facing_dir:
                theta = theta_noise
        if random_dir or noise_on_facing_dir:
            theta_half = theta / 2
            z, w = torch.sin(theta_half), torch.cos(theta_half)
            zeros = self.zeros1[:n_envs, None]
            q_ = torch.stack((zeros, zeros, z, w), -1)
            ref_link_tensor[:, :, :3]   = rotatepoint(q_, ref_link_tensor[:, :, :3])
            ref_link_tensor[:, :, 3:7]  = quatmultiply(q_, ref_link_tensor[:, :, 3:7])
            ref_link_tensor[:, :, 7:10] = rotatepoint(q_, ref_link_tensor[:, :, 7:10])
            ref_link_tensor[:, :, 10:]  = rotatepoint(q_, ref_link_tensor[:, :, 10:])
        
        ref_link_tensor[:, :, 0] += x
        if random_dir:
            ref_link_tensor[:, :, 1] += y
        if zero_vel:
            ref_link_tensor[:,-1,9] = -self.natural_dv

        self.released.index_fill_(0, env_ids, False)
        self.caught.index_fill_(0, env_ids, False)
        self.fallen.index_fill_(0, env_ids, False)
        self.ball_init_pose.index_copy_(0, env_ids, ref_link_tensor[:,-1])
        self.ball_vel.index_copy_(0, env_ids, ref_link_tensor[:, -1, 7:10])
        self.ball_pos.index_copy_(0, env_ids, ref_link_tensor[:, -1, :3])
        self.ball_height_before_release.index_fill_(0, env_ids, 0)
        self.foot_contacted[env_ids] = False
        self.foot_contact_vert[env_ids] = False
        self.pivot_foot.index_fill_(0, env_ids, -1)
        self.pivot_foot_set_at.index_fill_(0, env_ids, -1000)

        return ref_link_tensor, ref_joint_tensor
    
    def init_state(self, env_ids):
        ref_link_tensor, ref_joint_tensor = self._init_state(env_ids)
        ref_link_tensor = torch.cat((ref_link_tensor, self.court_init_pose[env_ids]), 1)
        return ref_link_tensor, ref_joint_tensor

    @staticmethod
    def _observe_goal(self, env_ids):
        if env_ids is None:
            root_tensor = self.root_tensor
            ball_tensor = self.ball_root_tensor
        else:
            root_tensor = self.root_tensor[env_ids]
            ball_tensor = self.ball_root_tensor[env_ids]
        return observe_shoot(root_tensor[:, 0], ball_tensor)

    def _observe(self, env_ids):
        ob = super()._observe(env_ids)
        g = self._observe_goal(self, env_ids)
        return torch.cat((ob, g), -1)
    
    def reward(self):
        ball_pos = self.ball_root_tensor[:, :3]
        ball_vel = self.ball_root_tensor[:, 7:10]

        key_fingers_left = self.left_finger_tips_with_palm
        key_fingers_right = self.right_finger_tips_with_palm
        left_ball_dist = torch.linalg.norm(self.link_pos[:, key_fingers_left, :3] - ball_pos.unsqueeze(1), ord=2, axis=-1)
        hold_ball_left = torch.all(left_ball_dist < self.BALL_RADIUS+0.01, 1)
        right_ball_dist = torch.linalg.norm(self.link_pos[:, key_fingers_right, :3] - ball_pos.unsqueeze(1), ord=2, axis=-1)
        hold_ball_right = torch.all(right_ball_dist < self.BALL_RADIUS+0.01, 1)
        hold_ball = torch.logical_and(hold_ball_left, hold_ball_right)
        hold_ball_single_hand = torch.logical_or(hold_ball_left, hold_ball_right)

        dv = (ball_vel - self.ball_vel)[:, 2]
        free_falling = (dv - self.natural_dv).abs_() < 1e-4
        launch = torch.logical_and(~self.caught, ~free_falling)
        self.caught[launch] = True
        release = torch.logical_and(self.caught, free_falling).logical_and_(~self.released)
        self.released[release] = True 

        self.ball_height_max[launch] = self.ball_pos[launch, 2]
        going_up = (ball_pos[:, 2] - self.ball_height_max) > 0.01
        
        key_fingers = self.palms
        key_fingers_rotation_axis = self.palms_rotation_axis
        finger_pos = self.link_pos[:, key_fingers, :3]
        face_dir = rotatepoint(self.link_orient[:, key_fingers], key_fingers_rotation_axis)
        e = face_dir.mul_(self.BALL_RADIUS).add_(finger_pos).sub_(ball_pos.unsqueeze(1)).square_().sum(-1)
        e.sqrt_()
        r_hold = e.mean(-1).mul_(-5).exp_()

        v = ball_vel[:, 2]
        not_fallen = ~self.fallen
        delta = v.square().add_(2*9.81*(ball_pos[:, 2]-self.BASKET_HEIGHT))
        invalid = delta < 1e-4
        t2 = v.add(delta.sqrt_()).div_(9.81)
        t = torch.where(invalid, (v>0)*v/9.81, t2)
        t = torch.minimum(t, (self.episode_length-self.lifetime)*self.step_time)
        p = ball_pos + ball_vel * t.unsqueeze(-1)
        p[:, 2] += (-0.5*9.81)*t.square()
        falling_down = not_fallen.logical_and(self.released).logical_and_(v<0).logical_and_(ball_pos[:, 2]<=self.BASKET_HEIGHT)
        frac = ((self.BASKET_HEIGHT-ball_pos[:, 2])/(self.ball_pos[:, 2]-ball_pos[:, 2]))
        interp = ball_pos + (self.ball_pos-ball_pos)*frac.nan_to_num_().clip_(min=0, max=1).unsqueeze_(-1)
        p = torch.where(falling_down.unsqueeze(-1), interp, p)
        p[:, 2] -= self.BASKET_HEIGHT
        p[:, 0] -= self.BASKET_POS
        p = torch.linalg.norm(p, ord=2, axis=-1)
        score = (self.ball_pos[:,2]>=self.BASKET_HEIGHT).logical_and_(falling_down).logical_and_(p<=self.BASKET_RADIUS-self.BALL_RADIUS)
        r_shoot = p.mul(-0.25).exp_()
        r_shoot[score] = 1.5
        self.ball_shoot_reward[falling_down] = r_shoot[falling_down]
        self.fallen[falling_down] = True
        r_shoot = torch.where(self.fallen, self.ball_shoot_reward, r_shoot)
        r_shoot *= self.ball_height_before_release>0
        
        d2b = (left_ball_dist-self.BALL_RADIUS).clip_(min=0).sum(-1) + (right_ball_dist-self.BALL_RADIUS).clip_(min=0).sum(-1)

        r_height = self.ball_height_before_release/self.BASKET_HEIGHT
        
        rew = torch.where(self.released,
            r_height + r_shoot,
            r_hold*0.5 + going_up*d2b.mul_(-20).exp_()
        )
        rew -= 0.1 * (~self.caught)
        
        shot = torch.sum(r_shoot > 1).item()
        tries = torch.sum(self.fallen).item()
        if tries and self.simulation_step > 0:
            self.info["shot_percent"].append(shot*1.0/tries)

        m = torch.logical_and(self.caught, ~self.released)
        up = torch.where(self.ball_height_before_release>0, hold_ball_single_hand, hold_ball)
        down = ball_pos[:, 2] < self.ball_height_before_release
        m.logical_and_(up.logical_or_(down))
        self.ball_height_before_release[m] = ball_pos[m, 2]
        self.ball_height_max.clip_(min=ball_pos[:,2])

        self.ball_pos.copy_(self.ball_root_tensor[:, :3])
        self.ball_vel.copy_(self.ball_root_tensor[:, 7:10])

        return rew.unsqueeze_(-1)

    def termination_check(self):
        fall = super().termination_check()
        not_shoot = (self.lifetime >= 40).logical_and_(~self.released)
        return fall.logical_or_(not_shoot)
    

@torch.jit.script
def observe_shoot(root_tensor: torch.Tensor, ball_tensor: torch.Tensor,
                  basket_xpos: float=ICCGANBall.BASKET_POS, basket_height: float=ICCGANBall.BASKET_HEIGHT
):    
    orient = root_tensor[:, 3:7]
    origin = root_tensor[:, :3].clone()
    origin[..., 2] = 0                                          # N x 3
    heading = heading_zup(orient)
    up_dir = torch.zeros_like(origin)
    up_dir[..., 2] = 1
    orient_inv = axang2quat(up_dir, -heading)                   # N x 4

    basket_pos = -origin
    basket_pos[..., 0] += basket_xpos
    basket_pos = rotatepoint(orient_inv, basket_pos)[..., :2]   # ignore the height, because the height is fixed
    ball_pos = rotatepoint(orient_inv, ball_tensor[..., :3]-origin)
    ball_vel = rotatepoint(orient_inv.unsqueeze_(1), ball_tensor[..., 7:13].view(-1, 2, 3))

    basket_pos /= 6.75
    ball_pos[:, :2] /= 6.75
    ball_pos[:, 2] /= basket_height*0.5
    ball_pos[:, 2] -= 1
    ball_vel /= 20
    heading /= 3.1415926535 # identify hoop orientation
    return torch.cat((ball_pos, ball_vel.view(-1, 6), basket_pos, heading.unsqueeze_(-1)), -1)


class ICCGANPass(ICCGANBall):
    GOAL_DIM = 9+3
    CAMERA_POS = 0, -8, 3.0

    def __init__(self, *args, **kwargs):
        self.pass_range = kwargs.get("pass_range", 1.05)
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.info["shot_percent"] = []

    def create_tensors(self):
        super().create_tensors()
        self.episode_length = 60

        n_envs = len(self.envs)
        self.caught = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.released = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.fallen = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.ball_pos = torch.zeros((n_envs,3), dtype=torch.float32, device=self.device)
        self.ball_vel = torch.zeros((n_envs,3), dtype=torch.float32, device=self.device)
        self.ball_height_before_release = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.ball_height_max = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.ball_shoot_reward = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.foot_contacted = torch.zeros((n_envs,2), dtype=torch.bool, device=self.device)
        self.pivot_foot = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.pivot_foot_set_at = torch.zeros((n_envs,), dtype=torch.int64, device=self.device)
        self.foot_contact_pos = torch.zeros((n_envs,2,4,3), dtype=torch.float32, device=self.device)
        self.foot_contact_vert = torch.zeros((n_envs,2,4,1), dtype=torch.bool, device=self.device)
        self.arange_2nenvs = torch.arange(n_envs*2, dtype=torch.int64, device=self.device)
        self.zeros1 = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.ref_dir_z = torch.zeros((3,), dtype=torch.float32, device=self.device)
        self.ref_dir_z[2] = 1

        rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], self.gym.get_actor_handle(self.envs[0], 0))

        self.palms = [rigid_body["RH:palm"], rigid_body["LH:palm"]]
        self.palms_rotation_axis = torch.tensor([
            [[0, 1, 0], [0, -1, 0]]
        ], dtype=torch.float, device=self.device)
        
        self.left_finger_tips_with_palm = sorted([i for n, i in rigid_body.items() if "LH:" in n and ("tip" in n or "palm" in n)])
        self.right_finger_tips_with_palm = sorted([i for n, i in rigid_body.items() if "RH:" in n and ("tip" in n or "palm" in n)])

        self.foot_links = [rigid_body["right_foot"], rigid_body["left_foot"]]
        self.foot_vertices = torch.tensor([[[
            [0.045+0.0885,  0.045, -0.0225-0.0275],
            [0.045+0.0885, -0.045, -0.0225-0.0275],
            [0.045-0.0885,  0.045, -0.0225-0.0275],
            [0.045-0.0885, -0.045, -0.0225-0.0275]
        ]]], dtype=torch.float32, device=self.device)
        self.wrist_links = [rigid_body["right_hand"], rigid_body["left_hand"]]

        self.target_tensor = torch.empty((len(self.envs), 3), dtype=torch.float32, device=self.device)

    def _init_state(self, env_ids, zero_vel=True, max_time=0, target_range=1.05):
        n_envs = len(env_ids)
        motion_ids, motion_times = self.ref_motion.sample(n_envs, max_time=max_time)
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)

        if zero_vel:
            ref_link_tensor[:,-1,9] = -self.natural_dv
        
        self.released.index_fill_(0, env_ids, False)
        self.caught.index_fill_(0, env_ids, False)
        self.fallen.index_fill_(0, env_ids, False)
        self.ball_init_pose.index_copy_(0, env_ids, ref_link_tensor[:,-1])
        self.ball_vel.index_copy_(0, env_ids, ref_link_tensor[:, -1, 7:10])
        self.ball_pos.index_copy_(0, env_ids, ref_link_tensor[:, -1, :3])
        self.ball_height_before_release.index_fill_(0, env_ids, 0)
        self.foot_contacted[env_ids] = False
        self.foot_contact_vert[env_ids] = False
        self.pivot_foot.index_fill_(0, env_ids, -1)
        self.pivot_foot_set_at.index_fill_(0, env_ids, -1000)
        self.ball_shoot_reward.index_fill_(0, env_ids, 0)

        # uniformly sample on a 2d ring range with inner radius 2.5 and outer radius 7.5
        # d = (u * (R**2 - r**2) + r**2)**(1/2)
        d = torch.rand((n_envs,), dtype=torch.float32, device=self.device).mul_(50).add_(6.25).sqrt_()
        theta = torch.rand((n_envs,), dtype=torch.float32, device=self.device).mul_(target_range*2).add_(-target_range) # +/-60 deg

        orient = ref_link_tensor[:, 0, 3:7]
        heading = heading_zup(orient)
        heading += theta
        x = torch.cos(heading)*d
        y = torch.sin(heading)*d
        z = torch.rand((n_envs,), dtype=torch.float32, device=self.device).mul_(0.3).add_(0.8) # 0.8-1.1

        self.target_tensor[env_ids, 0] = ref_link_tensor[:, 0, 0]+x
        self.target_tensor[env_ids, 1] = ref_link_tensor[:, 0, 1]+y
        self.target_tensor[env_ids, 2] = z

        return ref_link_tensor, ref_joint_tensor
    
    def init_state(self, env_ids):
        ref_link_tensor, ref_joint_tensor = self._init_state(env_ids, target_range=self.pass_range)
        ref_link_tensor = torch.cat((ref_link_tensor, self.court_init_pose[env_ids]), 1)
        return ref_link_tensor, ref_joint_tensor

    @staticmethod
    def _observe_goal(self, env_ids):
        if env_ids is None:
            root_tensor = self.root_tensor
            ball_tensor = self.ball_root_tensor
            target_tensor = self.target_tensor
        else:
            root_tensor = self.root_tensor[env_ids]
            ball_tensor = self.ball_root_tensor[env_ids]
            target_tensor = self.target_tensor[env_ids]
        return observe_pass(root_tensor[:, 0], ball_tensor, target_tensor)

    def _observe(self, env_ids):
        ob = super()._observe(env_ids)
        g = self._observe_goal(self, env_ids)
        return torch.cat((ob, g), -1)
    
    def reward(self):
        ball_pos = self.ball_root_tensor[:, :3]
        ball_vel = self.ball_root_tensor[:, 7:10]

        key_fingers_left = self.left_finger_tips_with_palm
        key_fingers_right = self.right_finger_tips_with_palm
        left_ball_dist = torch.linalg.norm(self.link_pos[:, key_fingers_left, :3] - ball_pos.unsqueeze(1), ord=2, axis=-1)
        hold_ball_left = torch.all(left_ball_dist < self.BALL_RADIUS+0.01, 1)
        right_ball_dist = torch.linalg.norm(self.link_pos[:, key_fingers_right, :3] - ball_pos.unsqueeze(1), ord=2, axis=-1)
        hold_ball_right = torch.all(right_ball_dist < self.BALL_RADIUS+0.01, 1)
        hold_ball = torch.logical_and(hold_ball_left, hold_ball_right)
        hold_ball_single_hand = torch.logical_or(hold_ball_left, hold_ball_right)

        dv = (ball_vel - self.ball_vel)[:, 2]
        free_falling = (dv - self.natural_dv).abs_() < 1e-4
        launch = torch.logical_and(~self.caught, ~free_falling)
        self.caught[launch] = True
        not_released = ~self.released
        release = torch.logical_and(self.caught, free_falling).logical_and_(not_released)
        just_released = torch.logical_and(release, not_released)
        self.released[release] = True 

        self.ball_height_max[launch] = self.ball_pos[launch, 2]
        going_up = (ball_pos[:, 2] - self.ball_height_max) > 0.01
        
        key_fingers = self.palms
        key_fingers_rotation_axis = self.palms_rotation_axis
        finger_pos = self.link_pos[:, key_fingers, :3]
        face_dir = rotatepoint(self.link_orient[:, key_fingers], key_fingers_rotation_axis)
        e = face_dir.mul_(self.BALL_RADIUS).add_(finger_pos).sub_(ball_pos.unsqueeze(1)).square_().sum(-1)
        e.sqrt_()
        r_hold = e.mean(-1).mul_(-5).exp_()

        tar_height = self.target_tensor[:, 2]
        tar_pos = self.target_tensor[:, :2]

        v = ball_vel[:, 2]
        not_fallen = ~self.fallen
        delta = v.square().add_(2*9.81*(ball_pos[:, 2]-tar_height))
        invalid = delta < 1e-4
        t_ = (self.episode_length-self.lifetime)*self.step_time

        t1 = (v+delta.sqrt()).div_(9.81).clip_(max=t_)
        t2 = (v-delta.sqrt()).div_(9.81).clip_(max=t_)
        t0 = ((v>0)*v/9.81).clip_(max=t_)
        p1 = ball_pos + ball_vel * t1.unsqueeze(-1)
        p2 = ball_pos + ball_vel * t2.unsqueeze(-1)
        dist1 = (p1[:,:2]-tar_pos).square_().sum(-1)
        dist2 = (p2[:,:2]-tar_pos).square_().sum(-1)
        t = torch.where(invalid, t0,
            torch.where(t2>0, torch.where(dist1 < dist2, t1, t2), t1))

        p = ball_pos + ball_vel * t.unsqueeze(-1)
        p[:, 2] += (-0.5*9.81)*t.square()
        released = not_fallen.logical_and(self.released)
        falling_down = released.logical_and(v<0).logical_and_(ball_pos[:, 2]<=tar_height)
        frac = ((tar_height-ball_pos[:, 2])/(self.ball_pos[:, 2]-ball_pos[:, 2]))
        interp = ball_pos + (self.ball_pos-ball_pos)*frac.nan_to_num_().clip_(min=0, max=1).unsqueeze_(-1)
        flying_up = released.logical_and(v>=0).logical_and_(ball_pos[:, 2]>=tar_height).logical_and_(self.ball_pos[:, 2]<=tar_height)
        falling_down = released.logical_and(v<0).logical_and_(ball_pos[:, 2]<=tar_height).logical_and_(
            (self.ball_pos[:, 2]>=tar_height).logical_or_(self.ball_vel[:, 2]>=0).logical_or_(just_released)
        )
        falling_down.logical_or_(flying_up)
        p = torch.where(falling_down.unsqueeze(-1), interp, p)
        p[:, 2] -= tar_height
        p[:, :2] -= tar_pos
        p = torch.linalg.norm(p, ord=2, axis=-1)

        threshold = 1.5*self.BALL_RADIUS
        score = ((self.target_tensor-ball_pos).square_().sum(-1).sqrt_() <= threshold).logical_or_(
            falling_down.logical_and(p <= threshold)
        )
            
        r_shoot = p.mul(-0.25).exp_()
        r_shoot[score] = 1.5
        
        root_pos = self.link_pos[:, 0, :2]
        too_far = (tar_pos-root_pos).square_().sum(-1).sqrt_()+threshold < (self.ball_pos[:, :2]-root_pos).square_().sum(-1).sqrt_()
        falling_down.logical_and_(too_far)
        
        fallen = falling_down.logical_or_(score)

        r_shoot = torch.where(t < 0, 
            torch.where(self.fallen, self.ball_shoot_reward, r_shoot),
            torch.maximum(self.ball_shoot_reward, r_shoot)
        )

        self.ball_shoot_reward[fallen] = torch.maximum(self.ball_shoot_reward[fallen], r_shoot[fallen])
        self.fallen[fallen] = True

        r_shoot = torch.where(self.fallen, self.ball_shoot_reward, r_shoot)
        r_shoot *= self.ball_height_before_release>0
        
        d2b = (left_ball_dist-self.BALL_RADIUS).clip_(min=0).sum(-1) + (right_ball_dist-self.BALL_RADIUS).clip_(min=0).sum(-1)
        r_height = self.ball_height_before_release/self.BASKET_HEIGHT

        rew = torch.where(self.released,
            r_height + r_shoot,
            r_hold*0.5 + going_up*d2b.mul_(-20).exp_()
        )

        rew -= 0.1 * (~self.caught)
        
        shot = torch.sum(r_shoot > 1).item()
        tries = torch.sum(self.fallen).item()
        if tries and self.simulation_step > 0: self.info["shot_percent"].append(shot*1.0/tries)

        reset = ~self.caught
        env_ids = self.ball_actor_ids[reset]
        if env_ids.numel() > 0:
            self.ball_root_tensor[reset] = self.ball_init_pose[reset]
            self.root_updated_actors.append(env_ids)
        
        m = torch.logical_and(self.caught, ~self.released)
        up = torch.where(self.ball_height_before_release>0, hold_ball_single_hand, hold_ball)
        down = ball_pos[:, 2] < self.ball_height_before_release
        m.logical_and_(up.logical_or_(down))
        self.ball_height_before_release[m] = ball_pos[m, 2]
        self.ball_height_max.clip_(min=ball_pos[:,2])

        self.ball_pos.copy_(self.ball_root_tensor[:, :3])
        self.ball_vel.copy_(self.ball_root_tensor[:, 7:10])

        if self.simulation_step:
            # prevent sliding
            p_foot_vert_ = rotatepoint(self.link_state_hist[-1, :, self.foot_links, None, 3:7], self.foot_vertices)
            p_foot_vert_.add_(self.link_state_hist[-1, :, self.foot_links, None, :3])    # N x 2 x 4 x 3

            p_foot_vert = rotatepoint(self.link_orient[:, self.foot_links, None, :], self.foot_vertices)
            p_foot_vert.add_(self.link_pos[:, self.foot_links, None, :])    # N x 2 x 4 x 3

            contacted_vertices_ = p_foot_vert[..., 2] < 0.01 # N x 2 x 4
            contacted_vertices_.logical_and_(p_foot_vert_[..., 2] < 0.01)

            contact = torch.any(contacted_vertices_, -1) # N x 2
            dp = (p_foot_vert - p_foot_vert_)[..., :2].abs_()
            contacted_vertices_.unsqueeze_(-1)
            foot_moved = torch.all((dp*contacted_vertices_ + (~contacted_vertices_)).view(-1, 2, 8) > 0.01, -1)
            foot_moved.logical_and_(contact)
            sliding = (self.lifetime>1).logical_and_(torch.any(foot_moved, -1))
            rew -= 0.25*sliding.to(torch.float)

        return rew.unsqueeze_(-1)

    def termination_check(self):
        fall = super().termination_check()
        not_shoot = (self.lifetime >= 40).logical_and_(~self.released)
        return fall.logical_or_(not_shoot)

    def update_viewer(self):
        super().update_viewer()
        self.gym.clear_lines(self.viewer)
        
        n_lines = 200
        phi = np.linspace(0, 2*np.pi, 20)
        theta = np.linspace(0, np.pi, 10)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        dx = 0.05 * (sin_phi[:, None] * cos_theta[None, :])
        dy = 0.05 * (sin_phi[:, None] * sin_theta[None, :])
        dz = 0.05 * cos_phi
        dx.shape = (-1, )
        dy.shape = (-1, )
        n_lines = len(dx)

        target = self.target_tensor.cpu().numpy()
        for i, (e, p) in enumerate(zip(self.envs, target)):
            l = np.stack([
                np.stack((p[0], p[1], p[2], p[0]+x, p[1]+y, p[2]+dz[i%len(dz)]))
                for i, (x, y) in enumerate(zip(dx, dy))
            ])
            self.gym.add_lines(self.viewer, e, n_lines, np.float32(l), np.float32([[0.,0.,1.] for _ in range(n_lines)]))

@torch.jit.script
def observe_pass(root_tensor: torch.Tensor, ball_tensor: torch.Tensor, target_tensor: torch.Tensor,
                 basket_height: float=ICCGANBall.BASKET_HEIGHT
):
    orient = root_tensor[:, 3:7]
    origin = root_tensor[:, :3].clone()
    origin[..., 2] = 0                                          # N x 3
    heading = heading_zup(orient)
    up_dir = torch.zeros_like(origin)
    up_dir[..., 2] = 1
    orient_inv = axang2quat(up_dir, -heading)

    target = rotatepoint(orient_inv, target_tensor-origin)
    target[:, :2] /= 6.75
    target[:, 2] /= basket_height*0.5
    target[:, 2] -= 1

    ball_pos = rotatepoint(orient_inv, ball_tensor[..., :3]-origin)
    ball_vel = rotatepoint(orient_inv.unsqueeze_(1), ball_tensor[..., 7:13].view(-1, 2, 3))
    ball_pos[:, :2] /= 6.75
    ball_pos[:, 2] /= basket_height*0.5
    ball_pos[:, 2] -= 1
    ball_vel /= 20
    return torch.cat((ball_pos, ball_vel.view(-1, 6), target), -1)


class ICCGANCatch(ICCGANBall):
    GOAL_DIM = 9
    CAMERA_POS = 0, -8, 3.0

    def __init__(self, *args, **kwargs):
        self.ingoing_range = kwargs.get("ingoing_range", 1.05) # 60deg
        super().__init__(*args, **kwargs)

    def create_tensors(self):
        super().create_tensors()
        self.episode_length = 60

        n_envs = len(self.envs)

        rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], self.gym.get_actor_handle(self.envs[0], 0))
        self.palms = [rigid_body["RH:palm"], rigid_body["LH:palm"]]
        self.palms_rotation_axis = torch.tensor([[[0, 1, 0], [0, -1, 0]]], dtype=torch.float, device=self.device)
        self.left_finger_tips_with_palm = sorted([i for n, i in rigid_body.items() if "LH:" in n and ("tip" in n or "palm" in n)])
        self.right_finger_tips_with_palm = sorted([i for n, i in rigid_body.items() if "RH:" in n and ("tip" in n or "palm" in n)])
        assert all([_ > 0 for _ in self.palms])
        assert all([_ > 0 for _ in self.left_finger_tips_with_palm])
        assert all([_ > 0 for _ in self.right_finger_tips_with_palm])

        self.foot_links = [rigid_body["right_foot"], rigid_body["left_foot"]]
        self.foot_vertices = torch.tensor([[[
            [0.045+0.0885,  0.045, -0.0225-0.0275],
            [0.045+0.0885, -0.045, -0.0225-0.0275],
            [0.045-0.0885,  0.045, -0.0225-0.0275],
            [0.045-0.0885, -0.045, -0.0225-0.0275]
        ]]], dtype=torch.float32, device=self.device)
        self.foot_contact_pos = torch.zeros((n_envs,2,4,3), dtype=torch.float32, device=self.device)
        self.foot_contact_vert = torch.zeros((n_envs,2,4,1), dtype=torch.bool, device=self.device)
        
        self.fallen = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.caught = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)

        self.ball_pos = torch.zeros((n_envs,3), dtype=torch.float32, device=self.device)
        self.ball_vel = torch.zeros((n_envs,3), dtype=torch.float32, device=self.device)

    def reset(self):
        super().reset()
        self.info["ball_hold"] = []

    @staticmethod
    def _observe_goal(self, env_ids):
        if env_ids is None:
            root_tensor = self.root_tensor
            ball_tensor = self.ball_root_tensor
        else:
            root_tensor = self.root_tensor[env_ids]
            ball_tensor = self.ball_root_tensor[env_ids]
        return observe_catch(root_tensor[:, 0], ball_tensor)

    def _observe(self, env_ids):
        ob = super()._observe(env_ids)
        g = self._observe_goal(self, env_ids)
        return torch.cat((ob, g), -1)
    
    def _init_state(self, env_ids, ingoing_range=1.05):
        n_envs = len(env_ids)
        motion_ids, motion_times = self.ref_motion.sample(n_envs, max_time=0.3)
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)

        # uniformly sample on a 2d ring range with inner radius 2.5 and outer radius 7.5
        # d = (u * (R**2 - r**2) + r**2)**(1/2)
        d = torch.rand((n_envs,), dtype=torch.float32, device=self.device).mul_(50).add_(6.25).sqrt_()
        theta = torch.rand((n_envs,), dtype=torch.float32, device=self.device).mul_(ingoing_range*2).add_(-ingoing_range) # +/-60 deg
        t = torch.rand((n_envs,), dtype=torch.float32, device=self.device).mul_(40*self.step_time).add_(10*self.step_time)

        x_ = torch.rand((n_envs,), dtype=torch.float32, device=self.device).add_(-0.5)
        y_ = torch.rand((n_envs,), dtype=torch.float32, device=self.device).add_(-0.5)
        z_ = torch.rand((n_envs,), dtype=torch.float32, device=self.device).mul_(0.6).add_(-0.1)

        orient = ref_link_tensor[:, 0, 3:7]
        heading = heading_zup(orient)
        heading += theta
        x = torch.cos(heading)*d
        y = torch.sin(heading)*d
        z = torch.rand((n_envs,), dtype=torch.float32, device=self.device).mul_(0.3).add_(0.8) # 0.8-1.1

        z_.add_(ref_link_tensor[:, 0, 2])

        dx = x_ - x
        dy = y_ - y
        dz = z_ - z

        vx = dx/t
        vy = dy/t
        # v*t-0.5gt^2 = dz
        vz = t.square().mul_(0.5*9.81).add_(dz).div_(t)

        ref_link_tensor[:, -1, 0] = ref_link_tensor[:, 0, 0] + x
        ref_link_tensor[:, -1, 1] = ref_link_tensor[:, 0, 1] + y 
        ref_link_tensor[:, -1, 2] = z 
        ref_link_tensor[:, -1, 7] = vx
        ref_link_tensor[:, -1, 8] = vy
        ref_link_tensor[:, -1, 9] = vz

        self.ball_pos[env_ids] = ref_link_tensor[:, -1, :3]
        self.ball_vel[env_ids] = ref_link_tensor[:, -1, 7:10]
        self.caught.index_fill_(0, env_ids, False)
        return ref_link_tensor, ref_joint_tensor

    def init_state(self, env_ids):
        ref_link_tensor, ref_joint_tensor = self._init_state(env_ids, ingoing_range=self.ingoing_range)
        self.fallen.index_fill_(0, env_ids, False)
        self.foot_contact_vert[env_ids] = False
        ref_link_tensor = torch.cat((ref_link_tensor, self.court_init_pose[env_ids]), 1)
        return ref_link_tensor, ref_joint_tensor

    def reward(self):
        ball_pos = self.ball_root_tensor[:, :3]
        ball_vel = self.ball_root_tensor[:, 7:10]

        v = self.ball_root_tensor[:, 9]

        dv = v - self.ball_vel[:, 2]
        free_falling = (dv - self.natural_dv).abs_() < 1e-4
        self.fallen[~free_falling] = True

        key_fingers_left = self.left_finger_tips_with_palm
        key_fingers_right = self.right_finger_tips_with_palm
        left_ball_dist = torch.linalg.norm(self.link_pos[:, key_fingers_left, :3] - ball_pos.unsqueeze(1), ord=2, axis=-1)
        right_ball_dist = torch.linalg.norm(self.link_pos[:, key_fingers_right, :3] - ball_pos.unsqueeze(1), ord=2, axis=-1)
        
        catch = (~free_falling).logical_and_(
            torch.any(left_ball_dist < self.BALL_RADIUS+0.01, 1)
            ).logical_and_(
            torch.any(right_ball_dist < self.BALL_RADIUS+0.01, 1))
        self.caught[catch] = True

        if self.simulation_step > 0:
            succ = torch.sum(catch).item()
            tries = torch.sum(self.fallen).item()
            if tries: self.info["ball_hold"].append(succ/tries)

        key_fingers = self.palms
        key_fingers_rotation_axis = self.palms_rotation_axis
        finger_pos = self.link_pos[:, key_fingers, :3]
        face_dir = rotatepoint(self.link_orient[:, key_fingers], key_fingers_rotation_axis)
        e_palms = face_dir.mul_(self.BALL_RADIUS).add_(finger_pos).sub_(ball_pos.unsqueeze(1)).square_().sum(-1)
        e_palms = e_palms.sqrt_().mean(-1)
        r_hold_palms = e_palms.mul(-1).exp_()*0.15 + e_palms.mul(-5).exp_()*0.35

        e_fingers = (left_ball_dist-self.BALL_RADIUS).clip_(min=0).sum(-1) + (right_ball_dist-self.BALL_RADIUS).clip_(min=0).sum(-1)
        r_hold_fingers = e_fingers.mul(-20).exp_()
        
        r_catch = r_hold_palms + r_hold_fingers

        ####### Foot Traveling ########
        # This is a simplified version of foot traveling detection,
        # where we only prefer the feet being fixed on the ground
        p_foot_vert = rotatepoint(self.link_orient[:, self.foot_links, None, :], self.foot_vertices)
        p_foot_vert.add_(self.link_pos[:, self.foot_links, None, :])    # N x 2 x 4 x 3
        contacted_vertices = p_foot_vert[..., 2] < 0.01 # N x 2 x 4

        contact = torch.any(contacted_vertices, -1) # N x 2
        dp = (p_foot_vert - self.foot_contact_pos)[..., :2].abs_()
        foot_moved = torch.all((dp*self.foot_contact_vert + (~self.foot_contact_vert)).view(-1, 2, 8) > 0.01, -1)
        foot_moved.logical_and_(contact)
        traveling = (self.lifetime>1).logical_and_(torch.any(foot_moved, -1)).to(torch.float)

        self.foot_contact_pos[contacted_vertices] = p_foot_vert[contacted_vertices] 
        self.foot_contact_vert[contacted_vertices] = True        
        ############## FOOT TRAVELING ############

        rew = r_catch - traveling*self.caught

        self.ball_pos.copy_(self.ball_root_tensor[:, :3])
        self.ball_vel.copy_(self.ball_root_tensor[:, 7:10])
        return rew.unsqueeze_(-1)
    
@torch.jit.script
def observe_catch(root_tensor: torch.Tensor, ball_tensor: torch.Tensor,
                  basket_height: float=ICCGANBall.BASKET_HEIGHT
):
    orient = root_tensor[:, 3:7]
    origin = root_tensor[:, :3].clone()
    origin[..., 2] = 0                                          # N x 3
    heading = heading_zup(orient)
    up_dir = torch.zeros_like(origin)
    up_dir[..., 2] = 1
    orient_inv = axang2quat(up_dir, -heading)

    ball_pos = rotatepoint(orient_inv, ball_tensor[..., :3]-origin)
    ball_vel = rotatepoint(orient_inv.unsqueeze_(1), ball_tensor[..., 7:13].view(-1, 2, 3))
    ball_pos[:, :2] /= 6.75
    ball_pos[:, 2] /= basket_height*0.5
    ball_pos[:, 2] -= 1
    ball_vel /= 20
    return torch.cat((ball_pos, ball_vel.view(-1, 6)), -1)


class ICCGANRebound(ICCGANBall):

    class BallPoseSampler:
        def __init__(self, env, capacity):
            self.env = env
            self.cache_len = 60
            self.cache = torch.empty((self.cache_len, len(env.envs), 13), dtype=torch.float, device=env.device)
            self.buffer_ball_init = torch.empty((capacity, 13), dtype=torch.float, device=env.device)
            self.buffer_ball_falling = torch.empty((capacity, 4), dtype=torch.float, device=env.device)
            self.cursor = 0
            self.capacity = self.buffer_ball_init.size(0)
            self.size = 0
            self.max_size = 0

        def record(self):
            self.cache[:-1] = self.cache[1:].clone()
            self.cache[-1] = self.env.ball_root_tensor
        
        def store(self, env_ids):
            if len(env_ids) > self.capacity:
                env_ids = env_ids[:self.capacity]
            last_frame = self.env.lifetime[env_ids].neg().add_(1)
            data = self.cache[(last_frame.clip(min=-20), env_ids)]

            n = len(env_ids)
            rem = self.capacity - self.cursor
            first = self.cursor
            if rem == n:
                last = None
                self.cursor = 0
                self.max_size = self.capacity
                self.buffer_ball_init[first:last] = data
                self.buffer_ball_falling[first:last, :2] = self.env.ball_root_tensor[env_ids, :2]
                self.buffer_ball_falling[first:last, 2:4] = self.env.ball_root_tensor[env_ids, 7:9]
            elif rem > n:
                last = self.cursor+n
                self.cursor += n
                self.max_size = max(self.max_size, self.cursor)
                self.buffer_ball_init[first:last] = data
                self.buffer_ball_falling[first:last, :2] = self.env.ball_root_tensor[env_ids, :2] 
                self.buffer_ball_falling[first:last, 2:4] = self.env.ball_root_tensor[env_ids, 7:9]
            else:
                n_rem = n - rem
                last = None
                self.cursor = n_rem
                self.max_size = self.capacity
                self.buffer_ball_init[:n_rem] = data[:n_rem]
                self.buffer_ball_init[first:last] = data[n_rem:]
                env_ids0 = env_ids[:n_rem]
                env_ids1 = env_ids[n_rem:]
                self.buffer_ball_falling[:n_rem, :2] = self.env.ball_root_tensor[env_ids0, :2] 
                self.buffer_ball_falling[:n_rem, 2:4] = self.env.ball_root_tensor[env_ids0, 7:9] 
                self.buffer_ball_falling[first:last, :2] = self.env.ball_root_tensor[env_ids1, :2] 
                self.buffer_ball_falling[first:last, 2:4] = self.env.ball_root_tensor[env_ids1, 7:9] 
            self.size = min(self.capacity, self.size+n)

        def sample(self, n):
            perm = torch.randperm(self.size)
            idx = perm[:n]
            return self.buffer_ball_init[idx], self.buffer_ball_falling[idx]

    def __init__(self, *args, **kwargs):
        self.GOAL_DIM = ICCGANShoot.GOAL_DIM + 1
        self._observe_goal = ICCGANShoot._observe_goal
            
        self.n_env_duplicates = 2
        if args:
            args = list(args)
            args[0] *= self.n_env_duplicates
        else:
            kwargs["n_envs"] *= self.n_env_duplicates
        class DummyBallPoseSampler:
            def record(self): pass
        self.ball_pose_sampler = DummyBallPoseSampler()
        super().__init__(*args, **kwargs)
        self.ball_pose_sampler = self.BallPoseSampler(self, len(self.envs)*2)
        self.dummy_actions = torch.zeros((self.n_envs1, self.act_dim), dtype=torch.float32, device=self.device)

        self.reset()
        self.eval()
        print("Generating samples...")
        reset_envs_task2 = self.reset_envs_task2
        self.reset_envs_task2 = self.reset_envs_task1
        while self.ball_pose_sampler.size < self.ball_pose_sampler.capacity:
            self.reset_done()
            self.step(self.dummy_actions[:(len(self.envs)-self.n_envs1)])
        self.reset_envs_task2 = reset_envs_task2
        print("{} Samples generated".format(self.ball_pose_sampler.size))

    def create_tensors(self):
        super().create_tensors()
        
        n_envs = len(self.envs)
        self.caught = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.released = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.fallen = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.collide = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.traveling = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self.ball_pos = torch.zeros((n_envs,3), dtype=torch.float32, device=self.device)
        self.ball_vel = torch.zeros((n_envs,3), dtype=torch.float32, device=self.device)
        self.ball_height_before_release = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.ball_height_max = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.ball_shoot_reward = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.foot_contacted = torch.zeros((n_envs,2), dtype=torch.bool, device=self.device)
        self.pivot_foot = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.pivot_foot_set_at = torch.zeros((n_envs,), dtype=torch.int64, device=self.device)
        self.foot_contact_pos = torch.zeros((n_envs,2,4,3), dtype=torch.float32, device=self.device)
        self.foot_contact_vert = torch.zeros((n_envs,2,4,1), dtype=torch.bool, device=self.device)
        self.arange_2nenvs = torch.arange(n_envs*2, dtype=torch.int64, device=self.device)
        self.zeros1 = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self.ref_dir_z = torch.zeros((3,), dtype=torch.float32, device=self.device)
        self.ref_dir_z[2] = 1

        rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], self.gym.get_actor_handle(self.envs[0], 0))
        self.palms = [rigid_body["RH:palm"], rigid_body["LH:palm"]]
        self.palms_rotation_axis = torch.tensor([[[0, 1, 0], [0, -1, 0]]], dtype=torch.float, device=self.device)
        self.left_finger_tips_with_palm = sorted([i for n, i in rigid_body.items() if "LH:" in n and ("tip" in n or "palm" in n)])
        self.right_finger_tips_with_palm = sorted([i for n, i in rigid_body.items() if "RH:" in n and ("tip" in n or "palm" in n)])
        assert all([_ > 0 for _ in self.palms])
        assert all([_ > 0 for _ in self.left_finger_tips_with_palm])
        assert all([_ > 0 for _ in self.right_finger_tips_with_palm])

        self.foot_links = [rigid_body["right_foot"], rigid_body["left_foot"]]
        self.foot_vertices = torch.tensor([[[
            [0.045+0.0885,  0.045, -0.0225-0.0275],
            [0.045+0.0885, -0.045, -0.0225-0.0275],
            [0.045-0.0885,  0.045, -0.0225-0.0275],
            [0.045-0.0885, -0.045, -0.0225-0.0275]
        ]]], dtype=torch.float32, device=self.device)
        

        self.n_envs = len(self.envs)
        self.n_envs1 = len(self.envs) - len(self.envs)//self.n_env_duplicates

        self.env_ids_task1 = torch.arange(0, self.n_envs1, device=self.device)
        self.env_ids_task2 = torch.arange(self.n_envs1, self.n_envs, device=self.device)
        nelem = self.state_hist.size(1)-self.state_hist.size(1)//self.n_env_duplicates
        self.state_hist_task2 = self.state_hist[:, nelem:]

        episode_length = torch.empty(self.n_envs, dtype=torch.int64, device=self.device)
        episode_length[:self.n_envs1] = 100
        episode_length[self.n_envs1:] = 60
        self.episode_length = episode_length
    
    def reset(self):
        super().reset()
        self.info["ball_hold"] = []
        self.info["caught_rate"] = []
        self.info["lifetime"] = self.lifetime[self.n_envs1:]
        if self.training:
            # set up sampler to perform sampling only for one environment component
            envs_all = self.envs
            self.envs = envs_all[self.n_envs1:]
            self.fetch_real_samples()
            self.envs = envs_all
        self.blender = [[] for _ in range(len(self.envs))]
    
    def reset_done(self):
        for i in torch.nonzero(self.done).view(-1).cpu().numpy():
            self.blender[i].clear()

        if not self.viewer_pause:
            if self.obs is None:
                self.reset_envs_task1(self.env_ids_task1)
                self.reset_envs_task2(self.env_ids_task2)
                self.obs = self.observe()
            else:
                env_ids = torch.nonzero(self.done[self.n_envs1:]).view(-1)
                if len(env_ids):
                    env_ids_task2 = env_ids + self.n_envs1
                    self.reset_envs_task2(env_ids_task2)
                    self.obs[env_ids] = self.observe(env_ids_task2)
        self.info["ob_seq_lens"] = self.ob_seq_lens[self.n_envs1:]
        self.info["reward_weights"] = self.reward_weights[self.n_envs1:]
        return self.obs, self.info
    
    def step(self, actions):
        actions = torch.cat((self.dummy_actions, actions))
        obs, rews, dones, info = super(ICCGANHumanoid, self).step(actions)

        if self.discriminators and self.training:
            seq_len = self.info["ob_seq_lens"]+1
            res = dict()
            for id, disc in self.discriminators.items():
                s = self.state_hist_task2[-disc.ob_horizon:]
                res[id] = observe_iccgan(s, seq_len, disc.key_links, disc.parent_link,
                    include_velocity=False, local_pos=disc.local_pos)
            info["disc_obs"] = res
            info["disc_obs_expert"] = self.fetch_real_samples()
        
        info["terminate"] = info["terminate"][self.n_envs1:]
        return obs, rews, dones[self.n_envs1:], info 

    def reset_envs_task1(self, env_ids):
        n_envs = len(env_ids)

        # uniformly sample on a 2d ring range with inner radius 2.5 and outer radius 7.5
        # d = (u * (R**2 - r**2) + r**2)**(1/2)
        d = torch.rand((n_envs,1), dtype=torch.float32, device=self.device).mul_(50).add_(6.25).sqrt_()
        d.neg_()
        theta = torch.rand((n_envs,1), dtype=torch.float32, device=self.device).mul_(3.56).add_(-1.78)
        y, x = torch.sin(theta)*d, self.BASKET_POS + torch.cos(theta)*d

        # .5^2 - BALL_RADIUS^2 = 0.237
        if self.training:
            d_ = torch.rand((n_envs,1), dtype=torch.float32, device=self.device).mul_(0.237).add_(self.BALL_RADIUS*self.BALL_RADIUS).sqrt_()
        else:
            d_ = self.BALL_RADIUS
        theta_ = torch.rand((n_envs,1), dtype=torch.float32, device=self.device).mul_(np.pi*2)
        y_, x_ = torch.sin(theta_)*d_, self.BASKET_POS + torch.cos(theta_)*d_

        dx = x_ - x
        dy = y_ - y
        dist = (dx.square()+dy.square()).sqrt_()

        z_ = self.BASKET_HEIGHT
        z = torch.rand((n_envs,1), dtype=torch.float32, device=self.device).mul_(2).add_(1.5) # 1.5 ~ 3.5

        dz = z_ - z
        t_min = dist.div(3) # max horizontal linear velocity is 3
        t_max = ((dist-dz)/(0.5*9.81)).sqrt_() # launch angle 45 deg
        t = torch.rand((n_envs,1), dtype=torch.float32, device=self.device).mul_(t_max-t_min).add_(t_min)

        vx = dx/t
        vy = dy/t
        # v*t-0.5gt^2 = dz
        vz = t.square().mul_(0.5*9.81).add_(dz).div_(t)
        
        ball_pose = torch.cat((
            x, y, z, self.ball_init_pose[env_ids, 3:7],
            vx, vy, vz,
            torch.rand((n_envs, 3), dtype=torch.float32, device=self.device).mul_(120).add_(-60)
        ), 1)

        self.ball_init_pose.index_copy_(0, env_ids, ball_pose)
        self.ball_vel.index_copy_(0, env_ids, ball_pose[:, 7:10])

        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)

        ref_link_tensor = torch.cat((ref_link_tensor[:, :-1], ball_pose.unsqueeze(1), self.court_init_pose[env_ids]), 1)

        self.root_tensor[env_ids] = ref_link_tensor[:, self.root_links]
        self.link_tensor[env_ids] = ref_link_tensor
        if self.action_tensor is None:
            self.joint_tensor[env_ids] = ref_joint_tensor
        else:
            self.joint_tensor[env_ids.unsqueeze(-1), self.actuated_dofs] = ref_joint_tensor
        
        self.root_updated_actors.append(self.actor_ids[env_ids].flatten())
        self.dof_updated_actors.append(self.actor_ids_having_dofs[env_ids].flatten())
        self.lifetime[env_ids] = 0

    def init_state_task2(self, env_ids):
        n_envs = len(env_ids)
        motion_ids, motion_times = self.ref_motion.sample(n_envs)
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)

        ball_pose, falling_state = self.ball_pose_sampler.sample(n_envs)

        falling_pos = falling_state[:, :2]
        falling_vel = falling_state[:, 2:4]
        heading = torch.atan2(falling_vel[:,1], falling_vel[:,0])

        noise = torch.rand((n_envs,), dtype=torch.float, device=self.device).mul_(2.10).add_(-1.05) #60 deg
        d = torch.rand((n_envs,), dtype=torch.float32, device=self.device)
        heading.add_(noise)

        p = falling_pos
        p[:, 0] += d*torch.cos(heading)
        p[:, 1] += d*torch.sin(heading)

        heading -= heading_zup(ref_link_tensor[:, 0, 3:7])
        heading += np.pi
        q_ = axang2quat(self.ref_dir_z, heading).unsqueeze_(1)

        ref_link_tensor[:,:,:2] -= ref_link_tensor[:,:1,:2].clone()
        ref_link_tensor[:,:,3:7] = quatmultiply(q_, ref_link_tensor[:,:,3:7])
        ref_link_tensor[:,:,7:10] = rotatepoint(q_, ref_link_tensor[:,:,7:10])
        ref_link_tensor[:,:,10:13] = rotatepoint(q_, ref_link_tensor[:,:,10:13])
        ref_link_tensor[:,:,:2] += p.unsqueeze_(1)

        ref_link_tensor[:, -1] = ball_pose

        self.released.index_fill_(0, env_ids, False)
        self.caught.index_fill_(0, env_ids, False)
        self.fallen.index_fill_(0, env_ids, False)
        self.collide.index_fill_(0, env_ids, False)
        self.traveling.index_fill_(0, env_ids, False)
        self.ball_init_pose.index_copy_(0, env_ids, ref_link_tensor[:,-1])
        self.ball_vel.index_copy_(0, env_ids, ref_link_tensor[:, -1, 7:10])
        self.ball_pos.index_copy_(0, env_ids, ref_link_tensor[:, -1, :3])
        self.ball_height_before_release.index_fill_(0, env_ids, 0)
        self.foot_contacted[env_ids] = False
        self.foot_contact_vert[env_ids] = False
        self.pivot_foot.index_fill_(0, env_ids, -1)
        self.pivot_foot_set_at.index_fill_(0, env_ids, -1000)

        ref_link_tensor = torch.cat((ref_link_tensor, self.court_init_pose[env_ids]), 1)
        return ref_link_tensor, ref_joint_tensor

    def reset_envs_task2(self, env_ids):
        self.init_state = self.init_state_task2
        super(ICCGANHumanoid, self).reset_envs(env_ids)
    
    def observe(self, env_ids=None):
        if env_ids is None and hasattr(self, "ref_motion") and not self.viewer_pause:
            env_ids_task1 = torch.nonzero(self.done[:self.n_envs1]).view(-1)
            if len(env_ids_task1): self.reset_envs_task1(env_ids_task1)
        return super().observe(env_ids)

    def _observe(self, env_ids):
        if env_ids is None:
            ob = super()._observe(self.env_ids_task2)
            g = self._observe_goal(self, self.env_ids_task2)
            self.ball_pose_sampler.record()
            pivot_foot = (self.pivot_foot[self.n_envs1:] - 0.5).div_(1.5).unsqueeze_(-1)
        else:
            ob = super()._observe(env_ids)
            g = self._observe_goal(self, env_ids)
            pivot_foot = (self.pivot_foot[env_ids] - 0.5).div_(1.5).unsqueeze_(-1)
        return torch.cat((ob, g, pivot_foot), -1)
    
    def reward(self):
        ball_pos = self.ball_root_tensor[:, :3]
        ball_vel = self.ball_root_tensor[:, 7:10]

        v = self.ball_root_tensor[:, 9]
        ball_height = ball_pos[:, 2]

        falling_down = ball_height<self.BASKET_HEIGHT

        dv = v - self.ball_vel[:, 2]
        free_falling = (dv - self.natural_dv).abs_() < 1e-4
        self.fallen[(~free_falling).logical_and_(falling_down)] = True
        self.collide[~free_falling] = True
        
        falling_down.logical_and_(v < 0).logical_and_(free_falling)
        self.info["falling_down"] = falling_down

        key_fingers_left = self.left_finger_tips_with_palm
        key_fingers_right = self.right_finger_tips_with_palm
        left_ball_dist = torch.linalg.norm(self.link_pos[:, key_fingers_left, :3] - ball_pos.unsqueeze(1), ord=2, axis=-1)
        right_ball_dist = torch.linalg.norm(self.link_pos[:, key_fingers_right, :3] - ball_pos.unsqueeze(1), ord=2, axis=-1)
        
        catch = (~free_falling).logical_and_(
            torch.any(left_ball_dist < self.BALL_RADIUS+0.01, 1)
            ).logical_and_(
            torch.any(right_ball_dist < self.BALL_RADIUS+0.01, 1))
        self.caught[catch] = True

        if self.simulation_step > 0:
            caught = torch.sum(self.caught[self.n_envs1:]).item()
            succ = torch.sum(catch[self.n_envs1:]).item()
            tries = torch.sum(self.fallen[self.n_envs1:]).item()
            if tries: self.info["ball_hold"].append(succ/tries)
            self.info["caught_rate"].append(caught/self.n_envs1)

        key_fingers = self.palms
        key_fingers_rotation_axis = self.palms_rotation_axis
        finger_pos = self.link_pos[:, key_fingers, :3]
        face_dir = rotatepoint(self.link_orient[:, key_fingers], key_fingers_rotation_axis)
        e_palms = face_dir.mul_(self.BALL_RADIUS).add_(finger_pos).sub_(ball_pos.unsqueeze(1)).square_().sum(-1)
        e_palms = e_palms.sqrt_().mean(-1)
        r_hold_palms = e_palms.mul(-1).exp_()*0.15 + e_palms.mul(-5).exp_()*0.35

        e_fingers = (left_ball_dist-self.BALL_RADIUS).clip_(min=0).sum(-1) + (right_ball_dist-self.BALL_RADIUS).clip_(min=0).sum(-1)
        r_hold_fingers = e_fingers.mul(-20).exp_()
        
        r_catch = r_hold_palms + r_hold_fingers
        

        ####### Foot Traveling ########
        p_foot_vert = rotatepoint(self.link_orient[:, self.foot_links, None, :], self.foot_vertices)
        p_foot_vert.add_(self.link_pos[:, self.foot_links, None, :])    # N x 2 x 4 x 3
        contacted_vertices = p_foot_vert[..., 2] < 0.01 # N x 2 x 4

        contact = torch.any(contacted_vertices, -1) # N x 2
        not_contact = ~contact
        all_feet_contacted = torch.all(contact, -1)
        foot0_contacted_only = torch.logical_and(contact[:, 0], not_contact[:, 1])
        foot1_contacted_only = torch.logical_and(contact[:, 1], not_contact[:, 0])
        activated = self.lifetime>1
        # activated.logical_or_(self.has_loaded)
        foot_drop = (~self.foot_contacted).logical_and_(contact).logical_and_(activated.unsqueeze_(-1))
        self.foot_contacted = contact

        dp = (p_foot_vert - self.foot_contact_pos)[..., :2].abs_()
        foot_moved = torch.all(torch.any(dp*self.foot_contact_vert + (~self.foot_contact_vert) > 0.01, -1), -1)
        foot_moved.logical_and_(contact)
        foot_moved_or_drop = torch.logical_or(foot_moved, foot_drop)

        cond0 = (self.pivot_foot == 0).logical_and_(foot_moved_or_drop[:, 0]) # pivot foot lifted up and now back down to the ground, the ball should be released
        cond1 = (self.pivot_foot == 1).logical_and_(foot_moved_or_drop[:, 1]) # same with cond0
        # two feet contacted previously or lifted up simultaneously
        # the ball should be released before any foot back if they were lifted up simultaneously
        # or if two feet moved
        # or if one foot lifted up (the other foot become pivot) and the other foot moves
        cond2 = (self.pivot_foot == 2).logical_and_(
            torch.any(foot_drop, -1).logical_or_(torch.all(foot_moved, -1)).logical_or_(
            torch.logical_and(not_contact[:, 0], foot_moved[:, 1])).logical_or_(
            torch.logical_and(not_contact[:, 1], foot_moved[:, 0])))
        traveling = cond0.logical_or(cond1).logical_or(cond2).logical_and_(~self.released)

        pivot_foot_undetected = (self.pivot_foot == -1)
        pivot_foot_notdecided = self.pivot_foot == 2
        pivot_foot_decided = (self.pivot_foot == 0).logical_or_(self.pivot_foot == 1)

        # if two feet on ground after holding the ball, let the pivot foot be undefined
        m2 = torch.logical_and(pivot_foot_undetected, all_feet_contacted)
        # if one foot on ground after holding the ball, set that foot as the pivot foot
        # if one foot lift up and pivot foot is undefined (two feet contacted the ground previously), set the contacted foot as pivot foot
        m0 = torch.logical_and(pivot_foot_undetected, foot0_contacted_only)
        m1 = torch.logical_and(pivot_foot_undetected, foot1_contacted_only)
        self.pivot_foot[m0] = 0
        self.pivot_foot[m1] = 1
        self.pivot_foot[m2] = 2

        m = m0.logical_or_(m1).logical_or_(m2)
        self.pivot_foot_set_at[m] = self.simulation_step

        # update already detected pivot foot
        # add 4 frame tolerance of two pivot feet detection
        m2 = (self.pivot_foot_set_at > self.simulation_step-3).logical_and_(all_feet_contacted).logical_and_(pivot_foot_decided)
        # update pivot if one foot lifts up or the other moves with two pivot feet
        #  0  1
        #  m  m  => 2
        # ~m ~m  => 2
        # ~m  m  => 0
        #  m ~c  => 0
        # ~m ~c  => 0
        #  m ~m  => 1
        # ~c  m  => 1
        # ~c ~m  => 1
        foot0_moved_only = (~foot_moved[:, 0]).logical_and_(foot_moved[:, 1]).logical_and_(all_feet_contacted)
        m0 = foot0_moved_only.logical_or(foot0_contacted_only).logical_and_(pivot_foot_notdecided)
        foot1_moved_only = (~foot_moved[:, 1]).logical_and_(foot_moved[:, 0]).logical_and_(all_feet_contacted)
        m1 = foot1_moved_only.logical_or(foot1_contacted_only).logical_and_(pivot_foot_notdecided)
        self.pivot_foot[m0] = 0
        self.pivot_foot[m1] = 1
        self.pivot_foot[m2] = 2

        m.logical_or_(m2)
        contacted_vertices.logical_and_((~self.foot_contact_vert).squeeze_(-1)).logical_and_(m[:,None,None])
        self.foot_contact_pos[contacted_vertices] = p_foot_vert[contacted_vertices] 
        self.foot_contact_vert[contacted_vertices] = True

        ball_not_hold = (~self.caught).logical_or_(self.released)
        self.pivot_foot[ball_not_hold] = -1
        self.foot_contact_vert[ball_not_hold] = False
        self.pivot_foot_set_at[ball_not_hold] = -1000
        self.traveling[traveling] = True
        ############## FOOT TRAVELING ############

        rew = r_catch - traveling.to(torch.float)

        self.ball_vel.copy_(self.ball_root_tensor[:, 7:10])
        rew = rew[self.n_envs1:]
        return rew.unsqueeze_(-1)

    def termination_check(self):
        term = super().termination_check()
        falling_down = self.info["falling_down"]
        if self.ball_pose_sampler.size < self.ball_pose_sampler.capacity:
            # generating samples
            term[:] = False
            env_ids = torch.nonzero(falling_down).view(-1)
        else:
            term[:self.n_envs1] = False
            env_ids = torch.nonzero(falling_down[:self.n_envs1]).view(-1)
        term[env_ids] = True
        if len(env_ids):
            self.ball_pose_sampler.store(env_ids)
        return term 

    def update_camera(self):
        tar_env = len(self.envs)-1 #min(len(self.envs)-1, 1) #len(self.envs)//4 + int(len(self.envs)**0.5)//2
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, self.envs[tar_env])
        base_pos = self.root_tensor[tar_env, -1, :3].cpu().detach()
        cam_pos = gymapi.Vec3(base_pos[0]+12, base_pos[1]-7, 4)
        self.cam_target = gymapi.Vec3(base_pos[0]+12, 0, 2.0)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)

