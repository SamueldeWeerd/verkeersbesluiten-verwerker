"""
Feature Matching Algorithms

This package contains different feature matching algorithms for various image types.
"""

from .base_matcher import BaseFeatureMatcher
from .schematic_matcher import SchematicMapMatcher
from .aerial_matcher import AerialImageryMatcher

__all__ = [
    'BaseFeatureMatcher',
    'SchematicMapMatcher', 
    'AerialImageryMatcher'
]
