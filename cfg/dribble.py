

env_cls = "ICCGANDribbleTarget"
env_params = dict(
    episode_length = 600,
    character_model = "assets/humanoid_hands.xml",
    motion_file = "assets/dribble.yaml",
    goal_reward_weight=[0.5, 0.2],
    ob_horizon = 2,

    goal_radius = 0.5,
    sp_lower_bound = 1.2,
    sp_upper_bound = 6,
    goal_timer_range = (60, 90),
    goal_timer_threshold = 0,
    goal_sp_mean = 3,
    goal_sp_std = 4,
    goal_sp_min = -1,
    goal_sp_max = 4,
)

training_params = dict(
    max_epochs =   200000,
    save_interval = 50000,
    terminate_reward = -25
)

replay = "lambda n: np.random.uniform(0.8, 1.2, size=(n,))"
discriminators = {
    "dribble/body": dict(
        key_links = [
            "pelvis",
            "torso", "head",
            "right_upper_arm", "right_lower_arm", # "right_hand", # "RH:palm",
            # "RH:thumb1", "RH:thumb2", "RH:thumb3", "RH:thumb_tip",
            # "RH:index1", "RH:index2", "RH:index3", "RH:index_tip",
            # "RH:middle1", "RH:middle2", "RH:middle3", "RH:middle_tip",
            # "RH:ring1", "RH:ring2", "RH:ring3", "RH:ring_tip",
            # "RH:pinky1", "RH:pinky2", "RH:pinky3", "RH:pinky_tip",

            "left_upper_arm", "left_lower_arm", # "left_hand", # "LH:palm",
            # "LH:thumb1", "LH:thumb2", "LH:thumb3", "LH:thumb_tip",
            # "LH:index1", "LH:index2", "LH:index3", "LH:index_tip",
            # "LH:middle1", "LH:middle2", "LH:middle3", "LH:middle_tip",
            # "LH:ring1", "LH:ring2", "LH:ring3", "LH:ring_tip",
            # "LH:pinky1", "LH:pinky2", "LH:pinky3", "LH:pinky_tip",

            "right_thigh", "right_shin", "right_foot", #"right_toe",
            "left_thigh", "left_shin", "left_foot", #"left_toe",
        ],
        parent_link = None,
        replay_speed = replay,
        weight = 0.2,
        motion_file = ["assets/dribble_nofinger.yaml", "assets/run.yaml"]
    ),
    "dribble/fingers": dict(
        key_links = [
            [
            # "right_upper_arm", "right_lower_arm", 
            "right_hand", # "RH:palm",
            "RH:thumb1", "RH:thumb2", "RH:thumb3", "RH:thumb_tip",
            "RH:index1", "RH:index2", "RH:index3", "RH:index_tip",
            "RH:middle1", "RH:middle2", "RH:middle3", "RH:middle_tip",
            "RH:ring1", "RH:ring2", "RH:ring3", "RH:ring_tip",
            "RH:pinky1", "RH:pinky2", "RH:pinky3", "RH:pinky_tip",
            ],
            [
            # "left_upper_arm", "left_lower_arm", 
            "left_hand", # "LH:palm",
            "LH:thumb1", "LH:thumb2", "LH:thumb3", "LH:thumb_tip",
            "LH:index1", "LH:index2", "LH:index3", "LH:index_tip",
            "LH:middle1", "LH:middle2", "LH:middle3", "LH:middle_tip",
            "LH:ring1", "LH:ring2", "LH:ring3", "LH:ring_tip",
            "LH:pinky1", "LH:pinky2", "LH:pinky3", "LH:pinky_tip",
            ]
        ],
        parent_link = ["right_lower_arm", "left_lower_arm"],
        replay_speed = replay,
        weight = 0.1
    )
}

