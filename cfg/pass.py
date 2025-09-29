
env_cls = "ICCGANPass"
env_params = dict(
    character_model = "assets/humanoid_hands.xml",
    motion_file = "assets/pass.yaml",
    goal_reward_weight=0.8,
    ob_horizon = 2
)

training_params = dict(
    max_epochs =   100000,
    save_interval = 50000,
    terminate_reward = -25
)

discriminators = {
    "pass/full": dict(
        key_links = [
            "pelvis", "torso", "head",
            "right_upper_arm", "right_lower_arm", "right_hand", # "RH:palm",
            "RH:thumb1", "RH:thumb2", "RH:thumb3", "RH:thumb_tip",
            "RH:index1", "RH:index2", "RH:index3", "RH:index_tip",
            "RH:middle1", "RH:middle2", "RH:middle3", "RH:middle_tip",
            "RH:ring1", "RH:ring2", "RH:ring3", "RH:ring_tip",
            "RH:pinky1", "RH:pinky2", "RH:pinky3", "RH:pinky_tip",

            "left_upper_arm", "left_lower_arm", "left_hand", # "LH:palm",
            "LH:thumb1", "LH:thumb2", "LH:thumb3", "LH:thumb_tip",
            "LH:index1", "LH:index2", "LH:index3", "LH:index_tip",
            "LH:middle1", "LH:middle2", "LH:middle3", "LH:middle_tip",
            "LH:ring1", "LH:ring2", "LH:ring3", "LH:ring_tip",
            "LH:pinky1", "LH:pinky2", "LH:pinky3", "LH:pinky_tip",

            "right_thigh", "right_shin", "right_foot", #"right_toe",
            "left_thigh", "left_shin", "left_foot", #"left_toe",
        ],
        parent_link = None,
    )
}

