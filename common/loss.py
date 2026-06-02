from matplotlib.pyplot import bone
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def mpjpe(predicted, target, return_joints_err=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    if not return_joints_err:
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    else:
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        from einops import rearrange
        errors = rearrange(errors, 'B T N -> N (B T)')
        errors = torch.mean(errors, dim=-1).cpu().numpy().reshape(-1) * 1000
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1)), errors

def _bone_parent_tensor(parents, device):
    if torch.is_tensor(parents):
        return parents.to(device=device, dtype=torch.long)
    return torch.as_tensor(parents, device=device, dtype=torch.long)

def bone_lengths(poses, parents, eps=1e-8):
    """
    Compute lengths for all skeleton edges defined by parent indices.
    Supports pose tensors shaped (..., joints, 3).
    """
    parents = _bone_parent_tensor(parents, poses.device)
    children = torch.nonzero(parents >= 0, as_tuple=False).squeeze(-1)
    parent_joints = parents[children]
    bone_vec = poses[..., children, :] - poses[..., parent_joints, :]
    return torch.sqrt(torch.sum(bone_vec * bone_vec, dim=-1).clamp(min=eps))

def bone_length_loss(predicted, target, parents):
    """
    Penalize predicted bone lengths that differ from the ground-truth skeleton.
    """
    assert predicted.shape == target.shape
    pred_lengths = bone_lengths(predicted, parents)
    target_lengths = bone_lengths(target, parents).detach()
    return F.smooth_l1_loss(pred_lengths, target_lengths)

def bone_symmetry_loss(predicted, parents, joints_left, joints_right):
    """
    Encourage mirrored left/right limbs to keep similar lengths.
    """
    if joints_left is None or joints_right is None or len(joints_left) == 0:
        return predicted.new_zeros(())

    parents_tensor = _bone_parent_tensor(parents, predicted.device)
    mirror = {}
    for left_joint, right_joint in zip(joints_left, joints_right):
        mirror[int(left_joint)] = int(right_joint)
        mirror[int(right_joint)] = int(left_joint)

    left_edges = []
    right_edges = []
    for left_child in joints_left:
        left_child = int(left_child)
        left_parent = int(parents_tensor[left_child].item())
        if left_parent < 0:
            continue
        right_child = mirror.get(left_child)
        right_parent = mirror.get(left_parent, left_parent)
        if right_child is None or right_parent < 0:
            continue
        if int(parents_tensor[right_child].item()) != right_parent:
            continue
        left_edges.append((left_child, left_parent))
        right_edges.append((right_child, right_parent))

    if len(left_edges) == 0:
        return predicted.new_zeros(())

    left_child = torch.as_tensor([edge[0] for edge in left_edges], device=predicted.device, dtype=torch.long)
    left_parent = torch.as_tensor([edge[1] for edge in left_edges], device=predicted.device, dtype=torch.long)
    right_child = torch.as_tensor([edge[0] for edge in right_edges], device=predicted.device, dtype=torch.long)
    right_parent = torch.as_tensor([edge[1] for edge in right_edges], device=predicted.device, dtype=torch.long)

    left_lengths = torch.norm(predicted[..., left_child, :] - predicted[..., left_parent, :], dim=-1)
    right_lengths = torch.norm(predicted[..., right_child, :] - predicted[..., right_parent, :], dim=-1)
    return F.smooth_l1_loss(left_lengths, right_lengths)

def bone_temporal_consistency_loss(predicted, parents):
    """
    Penalize frame-to-frame bone-length jitter inside a training clip.
    Expected training shape is (batch, frames, joints, 3).
    """
    if predicted.dim() < 4 or predicted.shape[1] < 2:
        return predicted.new_zeros(())
    lengths = bone_lengths(predicted, parents)
    return F.smooth_l1_loss(lengths[:, 1:], lengths[:, :-1])

def mpjpe_diffusion_all_min(predicted, target, mean_pos=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    if not mean_pos:
        t = predicted.shape[1]
        h = predicted.shape[2]
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        from einops import rearrange
        errors = rearrange(errors, 'b t h f n  -> t h b f n', )
        min_errors = torch.min(errors, dim=1, keepdim=False).values
        min_errors = min_errors.reshape(t, -1)
        min_errors = torch.mean(min_errors, dim=-1, keepdim=False)
        return min_errors
    else:
        t = predicted.shape[1]
        h = predicted.shape[2]
        mean_pose = torch.mean(predicted, dim=2, keepdim=False)
        target = target.unsqueeze(1).repeat(1, t, 1, 1, 1)
        errors = torch.norm(mean_pose - target, dim=len(target.shape) - 1)
        from einops import rearrange
        errors = rearrange(errors, 'b t f n  -> t b f n', )
        errors = errors.reshape(t, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        return errors

def mpjpe_diffusion_reproj(predicted, target, reproj_2d, target_2d):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """

    t = predicted.shape[1]
    h = predicted.shape[2]
    target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
    target_2d = target_2d.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
    errors = torch.norm(predicted - target, dim=len(target.shape)-1)
    errors_2d = torch.norm(reproj_2d - target_2d, dim=len(target_2d.shape) - 1)
    from einops import rearrange
    select_ind = torch.min(errors_2d, dim=2, keepdim=True).indices
    errors_select = torch.gather(errors, 2, select_ind)
    errors_select = rearrange(errors_select, 'b t h f n  -> t h b f n', )
    errors_select = errors_select.reshape(t, -1)
    errors_select = torch.mean(errors_select, dim=-1, keepdim=False)
    return errors_select

def mpjpe_diffusion(predicted, target, mean_pos=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    if not mean_pos:
        t = predicted.shape[1]
        h = predicted.shape[2]
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        from einops import rearrange
        errors = rearrange(errors, 'b t h f n  -> t h b f n', ).reshape(t, h, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        min_errors = torch.min(errors, dim=1, keepdim=False).values
        return min_errors
    else:
        t = predicted.shape[1]
        h = predicted.shape[2]
        mean_pose = torch.mean(predicted, dim=2, keepdim=False)
        target = target.unsqueeze(1).repeat(1, t, 1, 1, 1)
        errors = torch.norm(mean_pose - target, dim=len(target.shape) - 1)
        from einops import rearrange
        errors = rearrange(errors, 'b t f n  -> t b f n', )
        errors = errors.reshape(t, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        return errors

def mpjpe_diffusion_3dhp(predicted, target, valid_frame, mean_pos=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    from einops import rearrange

    valid_frame = valid_frame.squeeze(2)
    predicted = rearrange(predicted, 'b t h f n c  -> b f t h n c', )
    predicted_valid = predicted[valid_frame]
    target_valid = target[valid_frame]

    if not mean_pos:
        t = predicted_valid.shape[1]
        h = predicted_valid.shape[2]
        target_valid = target_valid.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1)
        errors = torch.norm(predicted_valid - target_valid, dim=len(target_valid.shape)-1)
        from einops import rearrange
        errors = rearrange(errors, 'f t h n  -> t h f n', ).reshape(t, h, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        min_errors = torch.min(errors, dim=1, keepdim=False).values
        return min_errors
    else:
        t = predicted_valid.shape[1]
        h = predicted_valid.shape[2]
        mean_pose = torch.mean(predicted_valid, dim=2, keepdim=False)
        target_valid = target_valid.unsqueeze(1).repeat(1, t, 1, 1)
        errors = torch.norm(mean_pose - target_valid, dim=len(target_valid.shape) - 1)
        from einops import rearrange
        errors = rearrange(errors, 'f t n -> t f n', )
        errors = errors.reshape(t, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        return errors


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))


def p_mpjpe_diffusion_all_min(predicted, target, mean_pos=False):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """


    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted.shape
    if not mean_pos:
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    else:
        predicted = torch.mean(predicted, dim=2, keepdim=False)
        target = target.unsqueeze(1).repeat(1, t_sz, 1, 1, 1)

    predicted = predicted.cpu().numpy().reshape(-1, j_sz, c_sz)
    target = target.cpu().numpy().reshape(-1, j_sz, c_sz)

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    if not mean_pos:
        target = target.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
        predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
        errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
        errors = errors.transpose(1, 2, 0, 3, 4)
        min_errors = np.min(errors, axis=1, keepdims=False)
        min_errors = min_errors.reshape(t_sz, -1)
        min_errors = np.mean(min_errors, axis=1, keepdims=False)
        return min_errors
    else:
        target = target.reshape(b_sz, t_sz, f_sz, j_sz, c_sz)
        predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, f_sz, j_sz, c_sz)
        errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
        errors = errors.transpose(1, 0, 2, 3)
        errors = errors.reshape(t_sz, -1)
        errors = np.mean(errors, axis=1, keepdims=False)
        return errors

def p_mpjpe_diffusion(predicted, target, mean_pos=False):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """


    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted.shape
    if not mean_pos:
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    else:
        predicted = torch.mean(predicted, dim=2, keepdim=False)
        target = target.unsqueeze(1).repeat(1, t_sz, 1, 1, 1)

    predicted = predicted.cpu().numpy().reshape(-1, j_sz, c_sz)
    target = target.cpu().numpy().reshape(-1, j_sz, c_sz)

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    if not mean_pos:
        target = target.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
        predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
        errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
        errors = errors.transpose(1, 2, 0, 3, 4).reshape(t_sz, h_sz, -1)
        errors = np.mean(errors, axis=2, keepdims=False)
        min_errors = np.min(errors, axis=1, keepdims=False)
        return min_errors
    else:
        target = target.reshape(b_sz, t_sz, f_sz, j_sz, c_sz)
        predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, f_sz, j_sz, c_sz)
        errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
        errors = errors.transpose(1, 0, 2, 3)
        errors = errors.reshape(t_sz, -1)
        errors = np.mean(errors, axis=1, keepdims=False)
        return errors

def p_mpjpe_diffusion_reproj(predicted, target, reproj_2d, target_2d):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    #assert predicted.shape == target.shape

    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted.shape

    target = target.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    target_2d = target_2d.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    errors_2d = torch.norm(reproj_2d - target_2d, dim=len(target_2d.shape) - 1)
    selec_ind = torch.min(errors_2d, dim=2, keepdims=True).indices


    predicted = predicted.cpu().numpy().reshape(-1, j_sz, c_sz)
    target = target.cpu().numpy().reshape(-1, j_sz, c_sz)

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t


    target = target.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
    predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
    errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    errors = torch.from_numpy(errors).cuda()
    errors_select = torch.gather(errors, 2, selec_ind)
    from einops import rearrange
    errors_select = rearrange(errors_select, 'b t h f n  -> t h b f n', )
    errors_select = errors_select.reshape(t_sz, -1)
    errors_select = torch.mean(errors_select, dim=-1, keepdim=False)
    errors_select = errors_select.cpu().numpy()

    return errors_select


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def mean_velocity_error_train(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    assert axis == 1
    velocity_predicted = predicted[:, 1:,:,:] - predicted[:, :-1,:,:]
    velocity_target = target[:, 1:, :, :] - target[:, :-1, :, :]

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))

def mean_velocity_error(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=axis)
    velocity_target = np.diff(target, axis=axis)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))
