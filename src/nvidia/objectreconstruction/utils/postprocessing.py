# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import sys
import shutil

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt, UsdUtils
import logging
logger = logging.getLogger(__name__)

def _resolve_texture_path(mtl_filepath: str, map_token: str) -> str:
    """Resolve MTL map_* path (often relative to the .mtl file)."""
    raw = map_token.strip().strip('"').strip("'")
    mtl_dir = os.path.dirname(os.path.abspath(mtl_filepath))
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(mtl_dir, raw))


def _resolve_texture_path_for_usd(mtl_filepath: str, map_token: str, usd_filepath: str) -> str:
    """
    Like _resolve_texture_path, but if that path is missing, try the texture basename next to the
    .mtl and the .usd. MTL sometimes lists e.g. ``0/material_0.png`` while the image is actually
    ``material_0.png`` beside the layer — USDZ localization then fails on the nested path.
    """
    primary = _resolve_texture_path(mtl_filepath, map_token)
    if os.path.isfile(primary):
        return primary
    raw = map_token.strip().strip('"').strip("'")
    base = os.path.basename(raw.replace("\\", "/"))
    if not base:
        return primary
    mtl_dir = os.path.dirname(os.path.abspath(mtl_filepath))
    usd_dir = os.path.dirname(os.path.abspath(usd_filepath))
    for folder in (mtl_dir, usd_dir):
        alt = os.path.normpath(os.path.join(folder, base))
        if os.path.isfile(alt):
            logger.warning(
                "map_Kd path %s not found; using %s for USD/USDZ.",
                primary,
                alt,
            )
            return alt
    return primary


def _texture_asset_path(tex_abs: str, usd_filepath: str, *, absolute: bool) -> Sdf.AssetPath:
    """Author texture path relative to the USD layer directory (portable / USDZ-friendly) unless absolute=True."""
    tex_abs = os.path.normpath(os.path.abspath(tex_abs))
    if absolute:
        return Sdf.AssetPath(tex_abs.replace("\\", "/"))
    usd_dir = os.path.normpath(os.path.dirname(os.path.abspath(usd_filepath)))
    tex_dir = os.path.dirname(tex_abs)
    bn = os.path.basename(tex_abs)
    flat = os.path.join(usd_dir, bn)

    # Texture beside the .usd: always author ./basename (never ./0/...) so USDZ localization matches disk.
    if tex_dir == usd_dir:
        return Sdf.AssetPath(bn.replace("\\", "/"))

    rel = os.path.relpath(tex_abs, start=usd_dir).replace("\\", "/")
    nested = os.path.normpath(os.path.join(usd_dir, *rel.split("/")))
    if not os.path.isfile(nested) and os.path.isfile(flat):
        rel = bn.replace("\\", "/")
    elif os.path.isfile(flat) and os.path.isfile(nested) and nested != os.path.normpath(flat):
        try:
            if os.path.samefile(nested, flat):
                rel = bn.replace("\\", "/")
        except OSError:
            pass
    return Sdf.AssetPath(rel)


def _obj_index(raw: str, count: int) -> int:
    """OBJ indices are 1-based; negative counts from end of list."""
    if not raw:
        return 0
    i = int(raw)
    if i > 0:
        return i - 1
    if i < 0:
        return count + i
    return 0


def _parse_face_corner(part: str, n_v: int, n_vt: int) -> tuple[int, int]:
    """Parse one corner of `f v/vt/vn` (vt/vn may be empty: `1//3`). Returns (v_idx, vt_idx) with -1 if no vt."""
    seg = part.split("/")
    vi = _obj_index(seg[0], n_v) if seg else 0
    vti = -1
    if len(seg) > 1 and seg[1] != "":
        vti = _obj_index(seg[1], n_vt)
    return vi, vti


def load_obj(obj_filepath: str) -> tuple[list[tuple[float, float, float]], list[list[tuple[int, int]]], list[tuple[float, float]]]:
    """
    Load OBJ geometry and per-face UV indices.

    Each face is a list of (vertex_index, texcoord_index) where texcoord_index is
    -1 if the corner has no vt in the face definition.
    """
    vertices: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    faces: list[list[tuple[int, int]]] = []

    with open(obj_filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                vertices.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("vt "):
                p = line.split()
                texcoords.append((float(p[1]), float(p[2])))
            elif line.startswith("f "):
                parts = line.split()
                corners: list[tuple[int, int]] = []
                for part in parts[1:]:
                    vi, vti = _parse_face_corner(part, len(vertices), len(texcoords))
                    corners.append((vi, vti))
                faces.append(corners)

    return vertices, faces, texcoords


def _parse_map_kd_path(line: str) -> str | None:
    """
    MTL map_Kd often includes modifiers before the filename, e.g.:
      map_Kd -mm 0 1 material_0.png
      map_Kd -clamp on textures/foo.png
    Taking the first token wrongly yields '0' or '-mm' and breaks USD/USDZ paths.
    """
    if not line.startswith("map_Kd"):
        return None
    parts = line.split(maxsplit=1)
    if len(parts) < 2:
        return None
    tokens = parts[1].strip().split()
    if not tokens:
        return None
    tex_ext = (".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp", ".exr", ".hdr", ".webp")
    # Prefer a token that looks like a filename with an image extension (e.g. map_Kd -mm 0 1 file.png).
    # Do not treat "0/material_0.png" modifiers as the path before the real filename token.
    for tok in reversed(tokens):
        tok = tok.strip()
        if tok.startswith("-"):
            continue
        t = tok.strip('"').strip("'")
        low = t.lower()
        if any(low.endswith(e) for e in tex_ext):
            return t
    for tok in reversed(tokens):
        tok = tok.strip()
        if tok.startswith("-"):
            continue
        t = tok.strip('"').strip("'")
        if "/" in t or "\\" in t:
            return t
    for tok in reversed(tokens):
        tok = tok.strip()
        if not tok.startswith("-"):
            return tok.strip('"').strip("'")
    return None


def load_mtl(mtl_filepath: str) -> dict:
    """Parse common MTL fields used for UsdPreviewSurface."""
    props: dict = {}
    with open(mtl_filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0] if parts else ""
            if key == "Kd" and len(parts) >= 4:
                props["Kd"] = tuple(map(float, parts[1:4]))
            elif key == "Ka" and len(parts) >= 4:
                props["Ka"] = tuple(map(float, parts[1:4]))
            elif key == "Ks" and len(parts) >= 4:
                props["Ks"] = tuple(map(float, parts[1:4]))
            elif line.startswith("map_Kd"):
                p = _parse_map_kd_path(line)
                if p:
                    props["map_Kd"] = p
    return props


def _build_st_facevarying(
    faces: list[list[tuple[int, int]]],
    texcoords: list[tuple[float, float]],
    n_vertices: int,
    flip_v: bool,
) -> list[Gf.Vec2f]:
    """
    One (u,v) per mesh corner, matching faceVertexIndices order.
    If a face omits vt but len(texcoords)==n_vertices, use the vertex index as UV index (some exporters).
    """
    out: list[Gf.Vec2f] = []
    per_vertex_uv = len(texcoords) == n_vertices and n_vertices > 0

    def corner_st(vi: int, vti: int) -> Gf.Vec2f:
        idx = vti
        if idx < 0 and per_vertex_uv and 0 <= vi < len(texcoords):
            idx = vi
        if 0 <= idx < len(texcoords):
            u, v = texcoords[idx]
            return Gf.Vec2f(u, (1.0 - v) if flip_v else v)
        return Gf.Vec2f(0.0, 0.0)

    for face in faces:
        for vi, vti in face:
            out.append(corner_st(vi, vti))
    return out


def _uv_coverage_stats(faces: list[list[tuple[int, int]]], texcoords: list, n_vertices: int) -> tuple[int, int]:
    """Count corners with explicit vt vs total corners."""
    per_vertex_uv = len(texcoords) == n_vertices and n_vertices > 0
    total = sum(len(f) for f in faces)
    with_uv = 0
    for face in faces:
        for vi, vti in face:
            ok = vti >= 0 and vti < len(texcoords)
            if not ok and per_vertex_uv and 0 <= vi < len(texcoords):
                ok = True
            if ok:
                with_uv += 1
    return with_uv, total


def build_preview_material(
    stage: Usd.Stage,
    mtl_filepath: str,
    props: dict,
    usd_filepath: str,
    *,
    absolute_texture_paths: bool,
) -> UsdShade.Material:
    """
    UsdPreviewSurface + UsdUVTexture + UsdPrimvarReader_float2 for diffuse.
    (UsdPreviewSurface has no texture inputs; a file path on it is ignored.)
    """
    mat_path = "/Model/Material"
    material = UsdShade.Material.Define(stage, mat_path)
    preview = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
    preview.CreateIdAttr("UsdPreviewSurface")

    kd = props.get("Kd", (0.82, 0.82, 0.82))
    ks = props.get("Ks", (0.05, 0.05, 0.05))
    preview.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(float(ks[0]), float(ks[1]), float(ks[2])))
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    map_kd = props.get("map_Kd")
    if map_kd:
        tex_abs = _resolve_texture_path_for_usd(mtl_filepath, map_kd, usd_filepath)
        if not os.path.isfile(tex_abs):
            logger.warning("map_Kd texture not found after resolution: %s", tex_abs)

        st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/stReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        uv_tex = UsdShade.Shader.Define(stage, f"{mat_path}/DiffuseTexture")
        uv_tex.CreateIdAttr("UsdUVTexture")
        uv_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            _texture_asset_path(tex_abs, usd_filepath, absolute=absolute_texture_paths)
        )
        uv_tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        uv_tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        uv_tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        # sRGB albedo for PNG/JPEG; omit or set "raw" for linear EXR etc.
        uv_tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        # MTL often sets Kd 0 0 0 when map_Kd carries all color; UsdUVTexture scale would zero the sample.
        if "Kd" in props:
            c = props["Kd"]
            if max(float(c[0]), float(c[1]), float(c[2])) > 1e-4:
                uv_tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                    Gf.Vec4f(float(c[0]), float(c[1]), float(c[2]), 1.0)
                )

        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(uv_tex.ConnectableAPI(), "rgb")
    else:
        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(kd[0]), float(kd[1]), float(kd[2]))
        )

    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    return material


def convert_obj_to_usd(
    obj_filepath: str,
    mtl_filepath: str,
    usd_filepath: str,
    flip_uv_v: bool = False,
    absolute_texture_paths: bool = False,
) -> bool:
    """
    Convert an OBJ+MTL file to a USD file with UsdPreviewSurface texturing.
    Stage metadata: Y-up, metersPerUnit 1.0, default prim /Model (Isaac Sim friendly).
    """
    stage = Usd.Stage.CreateNew(usd_filepath)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    xform = UsdGeom.Xform.Define(stage, "/Model")
    xform.AddTranslateOp().Set((0, 0, 0))
    xform.AddRotateXYZOp().Set((0, 0, 0))
    xform.AddScaleOp().Set((1, 1, 1))
    xform.GetPrim().CreateAttribute("userProperties:blenderName:object", Sdf.ValueTypeNames.String).Set("Model")
    xform.GetPrim().CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(
        ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    )

    vertices, faces, texcoords = load_obj(obj_filepath)
    if not vertices or not faces:
        logger.error("No vertices or faces found in OBJ file.")
        return False

    flat_indices = [vi for face in faces for vi, _vti in face]
    face_vertex_counts = [len(face) for face in faces]
    st_facevarying = _build_st_facevarying(
        faces, texcoords, n_vertices=len(vertices), flip_v=flip_uv_v
    )

    material_props = load_mtl(mtl_filepath)
    if material_props.get("map_Kd"):
        ok_uv, n_corners = _uv_coverage_stats(faces, texcoords, len(vertices))
        if n_corners and ok_uv < n_corners * 0.5:
            logger.error(
                f"Warning: only {ok_uv}/{n_corners} face corners have usable UVs; "
                "texture may look wrong. Check OBJ face lines use v/vt pairs.",
            )

    mesh = UsdGeom.Mesh.Define(stage, "/Model/Mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(flat_indices))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))
    mesh.CreateDoubleSidedAttr().Set(True)

    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    st = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray(st_facevarying))

    material = build_preview_material(
        stage,
        mtl_filepath,
        material_props,
        usd_filepath,
        absolute_texture_paths=absolute_texture_paths,
    )

    mesh_prim = mesh.GetPrim()
    binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    binding_api.Bind(material)

    mesh_xform = UsdGeom.Xformable(mesh)
    mesh_xform.AddTranslateOp().Set((0, 0, 0))
    mesh_xform.AddRotateXYZOp().Set((0, 0, 0))
    mesh_xform.AddScaleOp().Set((100, 100, 100))
    mesh.GetPrim().CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(
        ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    )

    stage.SetDefaultPrim(xform.GetPrim())
    stage.GetRootLayer().Save()
    return True



def export_usd_to_usdz(usd_filepath: str, usdz_filepath: str, materal_pic_path: str) -> bool:
    """
    Export a USD file and its associated material/texture files to USDZ format using UsdUtils.
    This packs the .usd along with any referenced .mtl or texture images into the USDZ archive.
    Args:
        usd_filepath: Path to the input .usd file
        usdz_filepath: Path to the output .usdz file
    Returns:
        True if export is successful, False otherwise.
    """
    try:
        os.makedirs("0", exist_ok=True)
        shutil.copy2(materal_pic_path,f"0/{os.path.basename(materal_pic_path)}")
        # USD 24+ / OpenUSD: CreateNewUsdzPackage(rootAssetPath, usdzOutputPath, ...).
        # It follows references and bundles dependencies; the older (usdzPath, list, list) API is gone.
        ok = UsdUtils.CreateNewUsdzPackage(
            Sdf.AssetPath(os.path.abspath(usd_filepath)), usdz_filepath
        )
        if not ok:
            logger.error("USDZ export failed (CreateNewUsdzPackage returned false).")
        shutil.rmtree("0")
        return bool(ok)
    except Exception as exc:
        logger.error("USDZ export error: %s", exc)
        return False


if __name__ == "__main__":
    obj_path = "/workspace/3d-object-reconstruction/data/samples/retail_item/textured_mesh.obj"
    mtl_path = "/workspace/3d-object-reconstruction/data/samples/retail_item/material.mtl"
    usd_path = "/workspace/3d-object-reconstruction/data/samples/retail_item/textured_mesh.usd"
    ok = convert_obj_to_usd(obj_path, mtl_path, usd_path)
    # Example: export USD to USDZ (if Ok)
    if ok:
        usdz_path = "/workspace/3d-object-reconstruction/data/samples/retail_item/textured_mesh.usdz"
        success = export_usd_to_usdz(usd_path, usdz_path,mtl_path)
        if success:
            print(f"Exported to USDZ: {usdz_path}")
        else:
            print(f"USDZ export failed.")