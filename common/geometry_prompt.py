import torch
from torch import nn
import torch.nn.functional as F

from common.camera import project_to_2d


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
        )

    def forward(self, x):
        return x + self.net(x)


class TemporalConvPose(nn.Module):
    """
    Lightweight TCN that predicts a coarse root-relative 3D pose from 2D joints.
    The optional 2D confidence channel is used when present; otherwise confidence
    is set to 1 for all joints.
    """

    def __init__(
        self,
        num_joints=17,
        in_features=3,
        hidden_channels=256,
        num_blocks=4,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.in_features = in_features

        self.input_proj = nn.Conv1d(num_joints * in_features, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
                for i in range(num_blocks)
            ]
        )
        self.output_proj = nn.Conv1d(hidden_channels, num_joints * 3, 1)

    def forward(self, input_2d):
        b, f, j, _ = input_2d.shape
        xy = input_2d[..., :2]
        if input_2d.shape[-1] > 2:
            confidence = input_2d[..., 2:3].clamp(0.0, 1.0)
        else:
            confidence = torch.ones(b, f, j, 1, device=input_2d.device, dtype=input_2d.dtype)

        x = torch.cat((xy, confidence), dim=-1)
        x = x.reshape(b, f, j * self.in_features).transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x).transpose(1, 2).reshape(b, f, j, 3)

        # Keep the coarse pose in the same root-relative coordinate system as
        # the diffusion target.
        return x - x[:, :, :1]


class GeometryPromptBuilder(nn.Module):
    """
    Converts the coarse TCN pose and camera geometry into per-joint prompt tokens.

    Prompt channels:
      projection error xy, projection error norm, camera ray xyz, absolute depth,
      confidence, coarse 3D xyz, projected 2D xy, observed 2D xy.
    """

    prompt_dim = 15

    def __init__(self, default_root_depth=2.0, depth_scale=10.0, pose_scale=1.0):
        super().__init__()
        self.default_root_depth = float(default_root_depth)
        self.depth_scale = float(depth_scale)
        self.pose_scale = float(pose_scale)

    @staticmethod
    def _match_batch(tensor, batch_size):
        if tensor is None:
            return None
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            repeat_shape = [batch_size] + [1] * (tensor.dim() - 1)
            return tensor.repeat(*repeat_shape)
        raise ValueError('Camera/root batch size must be 1 or match the pose batch size.')

    def _root_trajectory(self, coarse_pose, root_trajectory):
        b, f, _, _ = coarse_pose.shape
        root_trajectory = self._match_batch(root_trajectory, b)
        if root_trajectory is None:
            root_xy = torch.zeros(b, f, 1, 2, device=coarse_pose.device, dtype=coarse_pose.dtype)
            root_z = torch.full(
                (b, f, 1, 1),
                self.default_root_depth,
                device=coarse_pose.device,
                dtype=coarse_pose.dtype,
            )
            return torch.cat((root_xy, root_z), dim=-1)
        if root_trajectory.dim() == 3:
            root_trajectory = root_trajectory.unsqueeze(-2)
        return root_trajectory.to(device=coarse_pose.device, dtype=coarse_pose.dtype)

    def _camera_ray(self, input_xy, camera_params):
        b, f, j, _ = input_xy.shape
        if camera_params is None:
            ones = torch.ones(b, f, j, 1, device=input_xy.device, dtype=input_xy.dtype)
            return F.normalize(torch.cat((input_xy, ones), dim=-1), dim=-1)

        camera_params = self._match_batch(camera_params, b)
        camera_params = camera_params.to(device=input_xy.device, dtype=input_xy.dtype)
        focal = camera_params[:, :2].abs().clamp(min=1e-6).view(b, 1, 1, 2)
        center = camera_params[:, 2:4].view(b, 1, 1, 2)
        ray_xy = (input_xy - center) / focal
        ones = torch.ones(b, f, j, 1, device=input_xy.device, dtype=input_xy.dtype)
        return F.normalize(torch.cat((ray_xy, ones), dim=-1), dim=-1)

    def _project(self, absolute_pose, camera_params):
        b, f, j, _ = absolute_pose.shape
        if camera_params is None:
            z = absolute_pose[..., 2:].clamp(min=1e-3)
            return torch.clamp(absolute_pose[..., :2] / z, min=-1.0, max=1.0)

        camera_params = self._match_batch(camera_params, b)
        camera_params = camera_params.to(device=absolute_pose.device, dtype=absolute_pose.dtype)
        flat_pose = absolute_pose.reshape(b * f, j, 3)
        flat_camera = camera_params[:, None, :].expand(b, f, 9).reshape(b * f, 9)
        return project_to_2d(flat_pose, flat_camera).reshape(b, f, j, 2)

    def forward(self, input_2d, coarse_pose, camera_params=None, root_trajectory=None):
        input_xy = input_2d[..., :2]
        if input_2d.shape[-1] > 2:
            confidence = input_2d[..., 2:3].clamp(0.0, 1.0)
        else:
            confidence = torch.ones_like(input_xy[..., :1])

        root = self._root_trajectory(coarse_pose, root_trajectory)
        absolute_pose_raw = coarse_pose + root
        absolute_pose = torch.cat(
            (
                absolute_pose_raw[..., :2],
                absolute_pose_raw[..., 2:3].clamp(min=1e-3),
            ),
            dim=-1,
        )

        projected_2d = self._project(absolute_pose, camera_params)
        projection_error = torch.clamp(projected_2d - input_xy, min=-2.0, max=2.0)
        projection_error_norm = torch.norm(projection_error, dim=-1, keepdim=True).clamp(max=2.0)
        camera_ray = self._camera_ray(input_xy, camera_params)
        depth = torch.clamp(absolute_pose[..., 2:3] / self.depth_scale, min=-1.0, max=1.0)
        coarse_pose_norm = torch.clamp(coarse_pose / self.pose_scale, min=-2.0, max=2.0)

        return torch.cat(
            (
                projection_error,
                projection_error_norm,
                camera_ray,
                depth,
                confidence,
                coarse_pose_norm,
                projected_2d.clamp(min=-2.0, max=2.0),
                input_xy.clamp(min=-2.0, max=2.0),
            ),
            dim=-1,
        )
