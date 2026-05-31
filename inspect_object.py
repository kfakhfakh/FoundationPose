#!/usr/bin/env python3
"""Inspect a 3D object file and print its geometry, scene, material, texture,
color, bounds, and transform information.

Examples:
  python inspect_object.py /path/to/model.obj
  python inspect_object.py /path/to/model.glb --json
  python inspect_object.py /path/to/model.ply --save-json info.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import trimesh


def _to_serializable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    return value


def _safe_shape(x):
    try:
        return tuple(x.shape)
    except Exception:
        return None


def inspect_visual(visual) -> Dict[str, Any]:
    info: Dict[str, Any] = {"type": type(visual).__name__}

    if visual is None:
        info["has_visual"] = False
        return info

    info["has_visual"] = True
    if hasattr(visual, "kind"):
        info["kind"] = str(getattr(visual, "kind"))

    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is not None:
        try:
            info["vertex_colors_shape"] = _safe_shape(vertex_colors)
            info["vertex_colors_dtype"] = str(vertex_colors.dtype)
            info["has_vertex_colors"] = True
        except Exception:
            info["has_vertex_colors"] = True

    uv = getattr(visual, "uv", None)
    if uv is not None:
        try:
            info["uv_shape"] = _safe_shape(uv)
            info["has_uv"] = True
        except Exception:
            info["has_uv"] = True

    material = getattr(visual, "material", None)
    if material is not None:
        info["material_type"] = type(material).__name__
        for attr in [
            "name",
            "ambient",
            "diffuse",
            "specular",
            "glossiness",
            "roughness",
            "metallic",
            "baseColorFactor",
            "baseColorTexture",
            "image",
        ]:
            if hasattr(material, attr):
                value = getattr(material, attr)
                if attr == "image" and value is not None:
                    try:
                        info["material_image_size"] = value.size
                        info["material_image_mode"] = value.mode
                    except Exception:
                        info["material_image_present"] = True
                elif attr == "baseColorTexture" and value is not None:
                    info["baseColorTexture_type"] = type(value).__name__
                    if hasattr(value, "image") and getattr(value, "image", None) is not None:
                        try:
                            info["baseColorTexture_image_size"] = value.image.size
                            info["baseColorTexture_image_mode"] = value.image.mode
                        except Exception:
                            info["baseColorTexture_image_present"] = True
                else:
                    info[attr] = _to_serializable(value)

    return info


def inspect_geometry(name: str, geom) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "name": name,
        "type": type(geom).__name__,
        "is_watertight": bool(getattr(geom, "is_watertight", False)),
        "is_volume": bool(getattr(geom, "is_volume", False)),
        "is_empty": bool(getattr(geom, "is_empty", False)),
    }

    vertices = getattr(geom, "vertices", None)
    faces = getattr(geom, "faces", None)
    normals = getattr(geom, "vertex_normals", None)
    bounds = getattr(geom, "bounds", None)
    extents = getattr(geom, "extents", None)
    centroid = getattr(geom, "centroid", None)
    area = getattr(geom, "area", None)
    volume = getattr(geom, "volume", None)

    if vertices is not None:
        info["vertex_count"] = int(len(vertices))
        info["vertices_shape"] = _safe_shape(vertices)
        info["vertices_dtype"] = str(vertices.dtype)
    if faces is not None:
        info["face_count"] = int(len(faces))
        info["faces_shape"] = _safe_shape(faces)
        info["faces_dtype"] = str(faces.dtype)
    if normals is not None:
        info["vertex_normals_shape"] = _safe_shape(normals)
    if bounds is not None:
        info["bounds"] = _to_serializable(np.asarray(bounds))
    if extents is not None:
        info["extents"] = _to_serializable(np.asarray(extents))
    if centroid is not None:
        info["centroid"] = _to_serializable(np.asarray(centroid))
    if area is not None:
        info["surface_area"] = float(area)
    if volume is not None:
        info["volume"] = float(volume)

    info["visual"] = inspect_visual(getattr(geom, "visual", None))
    return info


def inspect_mesh_file(mesh_file: str) -> Dict[str, Any]:
    obj = trimesh.load(mesh_file, force=None, process=False)

    result: Dict[str, Any] = {
        "mesh_file": os.path.abspath(mesh_file),
        "exists": os.path.exists(mesh_file),
        "loaded_type": type(obj).__name__,
    }

    if isinstance(obj, trimesh.Scene):
        result["is_scene"] = True
        result["geometry_count"] = len(obj.geometry)
        result["geometry_names"] = list(obj.geometry.keys())
        result["scene_bounds"] = _to_serializable(np.asarray(obj.bounds)) if obj.bounds is not None else None
        result["scene_graph_nodes"] = list(obj.graph.nodes_geometry) if hasattr(obj.graph, "nodes_geometry") else []
        result["scene_info"] = {
            "camera_count": len(getattr(obj, "cameras", [])) if hasattr(obj, "cameras") else 0,
            "lights_count": len(getattr(obj, "lights", [])) if hasattr(obj, "lights") else 0,
        }

        geoms: List[Dict[str, Any]] = []
        # Dump the scene so node transforms are applied before measuring bounds.
        dumped = obj.dump(concatenate=False)
        if len(dumped) > 0:
            result["dumped_geometry_count"] = len(dumped)
            try:
                dumped_bounds = [g.bounds for g in dumped if getattr(g, "bounds", None) is not None]
                if len(dumped_bounds) > 0:
                    stacked = np.asarray(dumped_bounds, dtype=np.float64)
                    mins = stacked[:, 0, :].min(axis=0)
                    maxs = stacked[:, 1, :].max(axis=0)
                    result["dumped_scene_bounds"] = _to_serializable(np.stack([mins, maxs], axis=0))
            except Exception:
                pass

        for idx, geom in enumerate(dumped if len(dumped) > 0 else obj.geometry.values()):
            name = getattr(geom, "name", None) or f"geometry_{idx}"
            geoms.append(inspect_geometry(name, geom))
        result["geometries"] = geoms
        return result

    result["is_scene"] = False
    result.update(inspect_geometry(Path(mesh_file).name, obj))
    return result


def print_summary(info: Dict[str, Any]) -> None:
    print("=" * 80)
    print(f"File: {info['mesh_file']}")
    print(f"Loaded type: {info['loaded_type']}")
    print(f"Exists: {info['exists']}")
    print(f"Scene: {info['is_scene']}")

    if info.get("is_scene"):
        print(f"Geometry count: {info.get('geometry_count', 0)}")
        if info.get("scene_bounds") is not None:
            print(f"Scene bounds: {info['scene_bounds']}")
        for g in info.get("geometries", []):
            print("-" * 80)
            print(f"Geometry: {g.get('name')}")
            print(f"  Type: {g.get('type')}")
            print(f"  Vertices: {g.get('vertex_count')}")
            print(f"  Faces: {g.get('face_count')}")
            if g.get("extents") is not None:
                print(f"  Extents: {g['extents']}")
            if g.get("bounds") is not None:
                print(f"  Bounds: {g['bounds']}")
            if g.get("centroid") is not None:
                print(f"  Centroid: {g['centroid']}")
            if g.get("surface_area") is not None:
                print(f"  Surface area: {g['surface_area']}")
            if g.get("volume") is not None:
                print(f"  Volume: {g['volume']}")
            visual = g.get("visual", {})
            print(f"  Visual type: {visual.get('type')}")
            if visual.get("has_vertex_colors"):
                print(f"  Vertex colors: {visual.get('vertex_colors_shape')}")
            if visual.get("has_uv"):
                print(f"  UV: {visual.get('uv_shape')}")
            if visual.get("material_type"):
                print(f"  Material: {visual.get('material_type')}")
            if visual.get("material_image_size"):
                print(f"  Texture image: {visual.get('material_image_size')}")
            if visual.get("baseColorTexture_type"):
                print(f"  BaseColorTexture: {visual.get('baseColorTexture_type')}")
            if visual.get("baseColorTexture_image_size"):
                print(f"  BaseColorTexture image: {visual.get('baseColorTexture_image_size')}")
    else:
        print(f"Vertices: {info.get('vertex_count')}")
        print(f"Faces: {info.get('face_count')}")
        if info.get("extents") is not None:
            print(f"Extents: {info['extents']}")
        if info.get("bounds") is not None:
            print(f"Bounds: {info['bounds']}")
        if info.get("centroid") is not None:
            print(f"Centroid: {info['centroid']}")
        if info.get("surface_area") is not None:
            print(f"Surface area: {info['surface_area']}")
        if info.get("volume") is not None:
            print(f"Volume: {info['volume']}")
        visual = info.get("visual", {})
        print(f"Visual type: {visual.get('type')}")
        if visual.get("has_vertex_colors"):
            print(f"Vertex colors: {visual.get('vertex_colors_shape')}")
        if visual.get("has_uv"):
            print(f"UV: {visual.get('uv_shape')}")
        if visual.get("material_type"):
            print(f"Material: {visual.get('material_type')}")
        if visual.get("material_image_size"):
            print(f"Texture image: {visual.get('material_image_size')}")
        if visual.get("baseColorTexture_type"):
            print(f"BaseColorTexture: {visual.get('baseColorTexture_type')}")
        if visual.get("baseColorTexture_image_size"):
            print(f"BaseColorTexture image: {visual.get('baseColorTexture_image_size')}")

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a 3D object and print all available information.")
    parser.add_argument("mesh_file", type=str, help="Path to the object file (.obj, .glb, .ply, .stl, .dae, etc.)")
    parser.add_argument("--json", action="store_true", help="Print full JSON info")
    parser.add_argument("--save-json", type=str, default=None, help="Save full JSON info to a file")
    args = parser.parse_args()

    info = inspect_mesh_file(args.mesh_file)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(info), f, indent=2)

    if args.json:
        print(json.dumps(_to_serializable(info), indent=2))
    else:
        print_summary(info)


if __name__ == "__main__":
    main()
