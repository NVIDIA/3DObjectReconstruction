# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVIDIA namespace package.

This is a namespace package that allows multiple NVIDIA packages to coexist.
"""

__path__ = __import__('pkgutil').extend_path(__path__, __name__) 