# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Command-Line Interface for 3D Object Reconstruction.

This module provides command-line tools for running the reconstruction pipeline
and its individual components.
"""

from .main import main

__all__ = [
    'main',
] 