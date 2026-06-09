# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Optimizer module for 3D Object Reconstruction

This module provides the Theseus-based optimizer replacement for CUDA OptimizerGPU.
"""

from .theseus_optimizer import TheseusOptimizer

__all__ = ['TheseusOptimizer'] 