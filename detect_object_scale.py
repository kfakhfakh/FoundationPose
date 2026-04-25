#!/usr/bin/env python3
"""
Detect the scale/units of a 3D object mesh.
Simple script to analyze object dimensions and recommend scale factor.

Usage:
    python detect_object_scale.py /path/to/model.obj
    python detect_object_scale.py /path/to/model.ply
    python detect_object_scale.py /path/to/model.stl
"""

import sys
import os
import trimesh
import numpy as np


def detect_object_scale(mesh_path):
    """
    Load a 3D model and detect its scale/units.

    Args:
        mesh_path: Path to mesh file (.obj, .ply, .stl, etc.)

    Returns:
        dict: Contains scale info, dimensions, and recommendations
    """

    if not os.path.exists(mesh_path):
        print(f"❌ Error: File not found: {mesh_path}")
        return None

    try:
        # Load mesh
        mesh = trimesh.load(mesh_path, process=True)

        # Handle scenes
        if isinstance(mesh, trimesh.Scene):
            mesh_list = [geom for geom in mesh.geometry.values()]
            if len(mesh_list) > 1:
                mesh = trimesh.util.concatenate(mesh_list)
            else:
                mesh = mesh_list[0]

        # Get dimensions
        bounds = mesh.bounds
        extents = mesh.extents
        max_dim = max(extents)
        min_dim = min(extents)
        centroid = mesh.centroid

        print("\n" + "=" * 70)
        print(f"📊 OBJECT SCALE ANALYSIS: {os.path.basename(mesh_path)}")
        print("=" * 70)

        print(f"\n📐 DIMENSIONS:")
        print(f"   Extents (Width × Height × Depth):")
        print(f"     X: {extents[0]:10.6f}")
        print(f"     Y: {extents[1]:10.6f}")
        print(f"     Z: {extents[2]:10.6f}")
        print(f"   Max dimension: {max_dim:10.6f}")
        print(f"   Min dimension: {min_dim:10.6f}")

        print(f"\n📍 BOUNDING BOX:")
        print(f"   Min point: ({bounds[0][0]:10.6f}, {bounds[0][1]:10.6f}, {bounds[0][2]:10.6f})")
        print(f"   Max point: ({bounds[1][0]:10.6f}, {bounds[1][1]:10.6f}, {bounds[1][2]:10.6f})")
        print(f"   Centroid:  ({centroid[0]:10.6f}, {centroid[1]:10.6f}, {centroid[2]:10.6f})")

        print(f"\n🔍 MESH INFO:")
        print(f"   Vertices: {len(mesh.vertices)}")
        print(f"   Faces: {len(mesh.faces)}")
        print(f"   Volume: {mesh.volume:.6f}")

        # Determine scale (same thresholds as auto_detect_scale in run_inference_multi_class.py)
        print("\n" + "=" * 70)
        print("⚖️  SCALE DETECTION:")
        print("=" * 70)

        scale_info = {
            'max_dim': max_dim,
            'unit': None,
            'scale_factor': None,
            'description': None
        }

        if max_dim < 1000:
            if max_dim < 100:
                if max_dim < 10:
                    if max_dim < 1:
                        scale_info['unit'] = 'meters (m)'
                        scale_info['scale_factor'] = 1.0
                        scale_info['description'] = 'Standard human/furniture scale'
                        print(f"\n✓ Detected: METERS (STANDARD)")
                        print(f"  Size: {max_dim:.4f} m (max dimension)")
                        print(f"\n  RECOMMENDATION:")
                        print(f"  → NO SCALING NEEDED")
                        print(f"  → Model is already in meters (standard unit)")
                    else:
                        scale_info['unit'] = 'decimeters (dm)'
                        scale_info['scale_factor'] = 0.1
                        scale_info['description'] = 'Medium objects (books, shoes, bottles)'
                        print(f"\n✓ Detected: DECIMETERS")
                        print(f"  Size: {max_dim:.2f} dm (max dimension)")
                        print(f"\n  RECOMMENDATION:")
                        print(f"  → Use: --mesh_scale 0.1")
                        print(f"  → This converts decimeters to meters")
                else:
                    scale_info['unit'] = 'centimeters (cm)'
                    scale_info['scale_factor'] = 0.01
                    scale_info['description'] = 'Small objects (coins, toys, tools)'
                    print(f"\n✓ Detected: CENTIMETERS")
                    print(f"  Size: {max_dim:.2f} cm (max dimension)")
                    print(f"\n  RECOMMENDATION:")
                    print(f"  → Use: --mesh_scale 0.01")
                    print(f"  → This converts centimeters to meters")
            else:
                scale_info['unit'] = 'millimeters (mm)'
                scale_info['scale_factor'] = 0.001
                scale_info['description'] = 'Very small objects (tools, small parts)'
                print(f"\n✓ Detected: MILLIMETERS")
                print(f"  Size: {max_dim:.2f} mm (max dimension)")
                print(f"\n  RECOMMENDATION:")
                print(f"  → Use: --mesh_scale 0.001")
                print(f"  → This converts millimeters to meters")
        else:
            scale_info['unit'] = 'very large'
            scale_info['scale_factor'] = None
            scale_info['description'] = 'No scaling applied'
            print(f"\n⚠️  Detected: VERY LARGE SCALE")
            print(f"  Size: {max_dim:.2f} units (max dimension)")
            print(f"\n  RECOMMENDATION:")
            print(f"  → NO SCALING NEEDED")
            print(f"  → Model appears to be in very large units")

        print("\n" + "=" * 70)
        print("💡 USAGE EXAMPLES:")
        print("=" * 70)

        if scale_info['scale_factor'] and scale_info['scale_factor'] != 1.0:
            print(f"\nWith single object:")
            print(f"  python run_inference_multi_class.py \\")
            print(f"    --mesh_file {os.path.basename(mesh_path)} \\")
            print(f"    --mesh_scale {scale_info['scale_factor']}")
            print(f"\nWith multiple objects:")
            print(f"  python run_inference_multi_class.py \\")
            print(f"    --mesh_file model1.obj model2.ply \\")
            print(f"    --video_dir demo_data/")
            print(f"\nNote: Auto-scaling is enabled, so you may not need --mesh_scale!")
        else:
            print(f"\nNo scaling needed - just use:")
            print(f"  python run_inference_multi_class.py \\")
            print(f"    --mesh_file {os.path.basename(mesh_path)} \\")
            print(f"    --video_dir demo_data/")

        print("\n" + "=" * 70 + "\n")

        return scale_info

    except Exception as e:
        print(f"❌ Error loading mesh: {e}")
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <path_to_mesh_file>")
        print("\nExamples:")
        print(f"  python {sys.argv[0]} /path/to/model.obj")
        print(f"  python {sys.argv[0]} /path/to/model.ply")
        print(f"  python {sys.argv[0]} /mnt/c/Users/kfakh/Downloads/ycbv/models/obj_000005.ply")
        sys.exit(1)

    mesh_path = sys.argv[1]
    result = detect_object_scale(mesh_path)

    if result is None:
        sys.exit(1)
