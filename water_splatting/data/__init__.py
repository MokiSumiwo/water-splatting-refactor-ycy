"""Data manager extensions for water-splatting experiments."""

from .checkpointable_full_images_datamanager import (
    CheckpointableFullImageDatamanager,
    CheckpointableFullImageDatamanagerConfig,
)

__all__ = [
    "CheckpointableFullImageDatamanager",
    "CheckpointableFullImageDatamanagerConfig",
]
