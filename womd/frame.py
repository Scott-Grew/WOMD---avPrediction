import numpy as np


def rotation_matrix(heading):
    cosine, sine = np.cos(heading), np.sin(heading)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def positions_to_agent_frame(world_positions, frame_origin, frame_heading):
    centred = np.asarray(world_positions, dtype=np.float64) - np.asarray(frame_origin, dtype=np.float64)
    return centred @ rotation_matrix(frame_heading)


def positions_to_world_frame(agent_positions, frame_origin, frame_heading):
    rotated = np.asarray(agent_positions, dtype=np.float64) @ rotation_matrix(frame_heading).T
    return rotated + np.asarray(frame_origin, dtype=np.float64)


def directions_to_agent_frame(world_directions, frame_heading):
    return np.asarray(world_directions, dtype=np.float64) @ rotation_matrix(frame_heading)


def headings_to_agent_frame(world_headings, frame_heading):
    return wrap_to_pi(np.asarray(world_headings, dtype=np.float64) - frame_heading)


def wrap_to_pi(angles):
    return (np.asarray(angles, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi
