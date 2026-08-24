#!/usr/bin/env python3
"""Import 12d XML drawables into a File Geodatabase.

Usage:
  python main.py input.12dxml output.gdb
  python main.py input.12dxml output.gdb --spatial-reference 7855
  python main.py input.12dxml output.gdb --dry-run

The script reads the XML export from 12d Model and converts drawable geometry
stored as <data_2d>/<data_3d> coordinate lists into feature classes in a File
Geodatabase. It also creates a point-based annotation feature class from label
text values and their display specifications.

Default spatial reference is GDA2020 / MGA Zone 56 (WKID 7856).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Sequence, Tuple

try:
    import arcpy
except ImportError:  # pragma: no cover - ArcGIS Pro is required in production
    arcpy = None


def clean_name(name: str, max_length: int = 64) -> str:
    valid = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not valid:
        return "Drawable"
    if valid[0].isdigit():
        valid = "F_" + valid
    return valid[:max_length]


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def decode_xml(path: str) -> ET.Element:
    with open(path, "rb") as fh:
        data = fh.read()

    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        text = data.decode("utf-16", errors="strict")
    else:
        text = data.decode("utf-8-sig", errors="strict")

    return ET.fromstring(text)


def get_element_name(element: ET.Element) -> str:
    if element is None:
        return ""

    name = element.attrib.get("name")
    if name:
        return name.strip()

    for child in element:
        if local_name(child.tag).lower() == "name" and child.text:
            return child.text.strip()

    return ""


def parse_coordinate_text(text: str):
    rows = []
    if not text:
        return rows

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        cleaned = re.sub(r"[\t,;]+", " ", line)
        parts = cleaned.split()
        if len(parts) < 2:
            continue

        try:
            values = [float(p) for p in parts[:3]]
        except ValueError:
            continue

        rows.append(tuple(values))

    return rows


def parse_data_points(data_elem: ET.Element):
    points = []
    if data_elem is None:
        return points

    for child in data_elem:
        if local_name(child.tag).lower() != "p":
            continue

        text = (child.text or "").strip()
        if not text:
            continue

        values = parse_coordinate_text(text)
        if not values:
            continue

        for value_tuple in values:
            if len(value_tuple) == 2:
                points.append((float(value_tuple[0]), float(value_tuple[1])))
            elif len(value_tuple) >= 3:
                points.append((float(value_tuple[0]), float(value_tuple[1]), float(value_tuple[2])))

    return points


def extract_chainage_value(element: ET.Element):
    if element is None:
        return None

    for child in element:
        tag = local_name(child.tag).lower()
        if tag == "chainage":
            value = (child.text or "").strip()
            if value:
                try:
                    return float(value)
                except ValueError:
                    return value
        if tag in {"data_2d", "data_3d", "drawables", "children"}:
            nested = extract_chainage_value(child)
            if nested is not None:
                return nested
    return None


def build_feature_records(root: ET.Element):
    records = []

    for element in root.iter():
        children = list(element)
        if not children:
            continue

        data_children = [
            child for child in children
            if local_name(child.tag).lower() in {"data_2d", "data_3d"}
        ]

        if not data_children:
            continue

        for data_child in data_children:
            points = parse_data_points(data_child)
            if not points:
               continue

            attributes = flatten_attribute_values(element)
            name = get_element_name(element) or local_name(element.tag)
            attributes["Name"] = name
            attributes["ParentTag"] = local_name(element.tag)
            chainage = extract_chainage_value(element)
            if chainage is not None:
               attributes["Chainage"] = coerce_scalar(chainage)
            else:
               attributes["Chainage"] = None

            records.append(
               {
                   "name": name,
                   "parent_tag": local_name(element.tag),
                   "points": points,
                   "dimension": "3D" if local_name(data_child.tag).lower() == "data_3d" else "2D",
                   "chainage": chainage,
                   "attributes": attributes,
               }
            )

    return records


def coerce_scalar(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value

    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        if "." in text or "e" in lowered:
            return float(text)
        return int(text)
    except ValueError:
        return text


def flatten_attribute_values(element: ET.Element, prefix: str = "") -> dict:
    values = {}
    if element is None:
        return values

    for child in element:
        tag = local_name(child.tag)
        tag_lower = tag.lower()

        if tag_lower in {"data_2d", "data_3d"}:
            continue

        if tag_lower in {"attributes", "group"}:
            nested = flatten_attribute_values(child, prefix)
            values.update(nested)
            continue

        if tag_lower in {"text", "real", "int", "bool", "string", "date"}:
            name_elem = child.find("name")
            value_elem = child.find("value")
            if name_elem is None and value_elem is None:
                continue
            name = (name_elem.text if name_elem is not None else "").strip() or (child.attrib.get("name") or "").strip()
            value = value_elem.text if value_elem is not None else (child.text or "").strip()
            if not name:
                continue
            values[prefix + name] = coerce_scalar(value)
            continue

        value = (child.text or "").strip()
        if value:
            values[prefix + tag] = coerce_scalar(value)

    return values


def build_string_super_records(root: ET.Element):
    records = []

    for element in root.iter():
        if local_name(element.tag).lower() != "string_super":
            continue

        name = get_element_name(element)
        if not name:
            name_elem = element.find("name")
            if name_elem is not None:
                name = (name_elem.text or "").strip()
        if not name:
            name = local_name(element.tag)

        points = []
        dimension = "2D"
        for child in element:
            tag = local_name(child.tag).lower()
            if tag in {"data_2d", "data_3d"}:
                points = parse_data_points(child)
                dimension = "3D" if tag == "data_3d" else "2D"
                break

        if not points:
            continue

        attributes = flatten_attribute_values(element)
        attributes["Name"] = name
        chainage = attributes.get("chainage")
        if chainage is None:
            chainage = extract_chainage_value(element)
        if chainage is not None:
            attributes["Chainage"] = coerce_scalar(chainage)
        else:
            attributes["Chainage"] = None

        records.append(
            {
                "name": name,
                "parent_tag": local_name(element.tag),
                "points": points,
                "dimension": dimension,
                "attributes": attributes,
                "chainage": chainage,
            }
        )

    return records


def parse_tin_float(value: str) -> float:
    value = (value or "").strip()
    if not value:
        return 0.0
    if value.lower().startswith("0x"):
        return float.fromhex(value)
    return float(value)


def parse_full_tin_points(full_tin_elem: ET.Element):
    points = []
    points_elem = None
    for child in full_tin_elem:
        if local_name(child.tag).lower() == "points":
            points_elem = child
            break

    if points_elem is None:
        return points

    for p_elem in points_elem:
        if local_name(p_elem.tag).lower() != "p":
            continue

        text = (p_elem.text or "").strip()
        parts = text.split()
        if len(parts) < 3:
            continue

        try:
            x = parse_tin_float(parts[0])
            y = parse_tin_float(parts[1])
            z = parse_tin_float(parts[2])
            points.append((x, y, z))
        except ValueError:
            continue

    return points


def build_tin_records(root: ET.Element):
    records = []

    for element in root.iter():
        if local_name(element.tag).lower() != "full_tin":
            continue

        name = ""
        points = []

        for child in element:
            tag = local_name(child.tag).lower()
            if tag == "name":
                name = (child.text or "").strip() or name
            elif tag == "points":
                points = parse_full_tin_points(element)

        if name and points:
            records.append({"name": name, "points": points})

    return records


def extract_chainage_records(root: ET.Element):
    records = []

    for alignment in root.iter():
        if local_name(alignment.tag).lower() != "string_super_alignment":
            continue

        alignment_name = ""
        for child in alignment:
            tag = local_name(child.tag).lower()
            if tag == "name":
                alignment_name = (child.text or "").strip() or alignment_name

        if not alignment_name:
            alignment_name = alignment.attrib.get("name", "Unknown")

        for data_tag in ("horizontal_data", "vertical_data"):
            data_elem = None
            for child in alignment:
                if local_name(child.tag).lower() == data_tag:
                    data_elem = child
                    break

            if data_elem is None:
                continue

            chainage = None
            breakline = ""
            colour = ""
            style = ""
            points = []

            for child in data_elem:
                tag = local_name(child.tag).lower()
                if tag == "chainage":
                    chainage = (child.text or "").strip()
                elif tag == "breakline":
                    breakline = (child.text or "").strip()
                elif tag == "colour":
                    colour = (child.text or "").strip()
                elif tag == "style":
                    style = (child.text or "").strip()
                elif tag in {"data_2d", "data_3d"}:
                    points = parse_data_points(child)

            if chainage is None:
                continue

            records.append(
                {
                    "alignment_name": alignment_name,
                    "chainage": chainage,
                    "breakline": breakline,
                    "colour": colour,
                    "style": style,
                    "points": points,
                    "dimension": "3D" if any(len(pt) == 3 for pt in points) else "2D",
                }
            )

    return records


def properties_to_dict(element: ET.Element):
    props = {}
    if element is None:
        return props

    for child in element:
        tag = local_name(child.tag)
        value = (child.text or "").strip()
        props[tag] = value

    return props


def build_annotation_records(root: ET.Element):
    records = []

    for element in root.iter():
        if local_name(element.tag).lower() not in {"horizontal_tangent", "horizontal_interval", "horizontal_name", "vertical_tangent", "vertical_interval", "vertical_name", "string_name", "label"}:
            continue

        text_value = None
        points = []
        spec = {}

        for child in element:
            tag = local_name(child.tag).lower()
            if tag == "data_2d":
                points = parse_data_points(child)
            elif tag == "vertex_text_value":
                text_value = (child.text or "").strip()
            elif tag == "vertex_text_data":
                text_value = (child.text or "").strip() or text_value
            elif tag == "vertex_annotate_data":
                spec.update(properties_to_dict(child.find("properties")))
            elif tag == "symbol_data":
                spec.update(properties_to_dict(child.find("properties")))

        if not text_value or not points:
            # support direct annotation objects with name + data_2d but no vertex_text_value
            if not points:
                continue
            text_value = get_element_name(element) or local_name(element.tag)

        x, y = points[0][:2]
        records.append(
            {
                "name": get_element_name(element) or text_value,
                "parent_tag": local_name(element.tag),
                "text": text_value,
                "x": x,
                "y": y,
                "chainage": extract_chainage_value(element),
                "rotation": float(spec.get("angle", spec.get("rotation", "0")) or 0),
                "font_size": float(spec.get("worldsize", spec.get("size", "10")) or 10),
                "font_name": spec.get("textstyle", spec.get("style", "Arial")) or "Arial",
                "text_colour": spec.get("text_colour", spec.get("colour", "black")) or "black",
                "justification": spec.get("justify", "bottom-left") or "bottom-left",
                "offset": float(spec.get("offset", "0") or 0),
                "raise": float(spec.get("raise", "0") or 0),
            }
        )

    return records


def geometry_type_for(points: Sequence[Tuple[float, ...]]) -> str:
    if len(points) == 1:
        return "POINT"
    if len(points) > 2 and points[0] == points[-1]:
        return "POLYGON"
    return "POLYLINE"


def valid_point_values(points: Sequence[Tuple[float, ...]]) -> bool:
    for pt in points:
        if len(pt) < 2:
            return False
        for value in pt[:3]:
            if not math.isfinite(value):
                return False
            if abs(value) > 1e15:
                return False
    return bool(points)


def build_geometry_for(points: Sequence[Tuple[float, ...]], spatial_reference):
    if not valid_point_values(points):
        raise ValueError("Invalid coordinate values encountered")

    if len(points) == 1:
        x, y = points[0][:2]
        if len(points[0]) == 3:
            return arcpy.PointGeometry(arcpy.Point(x, y, points[0][2]), spatial_reference)
        return arcpy.PointGeometry(arcpy.Point(x, y), spatial_reference)

    array = arcpy.Array()
    for pt in points:
        if len(pt) == 3:
            array.add(arcpy.Point(pt[0], pt[1], pt[2]))
        else:
            array.add(arcpy.Point(pt[0], pt[1]))

    if len(points) > 2 and points[0] == points[-1]:
        return arcpy.Polygon(array, spatial_reference)

    return arcpy.Polyline(array, spatial_reference)


def create_file_gdb(path: str) -> None:
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if not os.path.exists(path):
        arcpy.management.CreateFileGDB(folder, os.path.basename(path))


def add_field_if_not_exists(table_path: str, field_name: str, field_type: str, **kwargs) -> bool:
    """Add a field to table_path only if it does not already exist.
    Returns True if the field was created, False if it already existed.
    """
    existing = [f.name.upper() for f in arcpy.ListFields(table_path)] if arcpy is not None else []
    if field_name.upper() in existing:
        return False
    arcpy.management.AddField(table_path, field_name, field_type, **kwargs)
    return True


def ensure_feature_class(gdb_path: str, name: str, geometry_type: str, spatial_reference) -> str:
    fc_path = os.path.join(gdb_path, name)
    if arcpy.Exists(fc_path):
        arcpy.management.Delete(fc_path)

    arcpy.management.CreateFeatureclass(
        gdb_path,
        name,
        geometry_type,
        spatial_reference=spatial_reference,
    )

    add_field_if_not_exists(fc_path, "Name", "TEXT", field_length=255)
    add_field_if_not_exists(fc_path, "ParentTag", "TEXT", field_length=255)
    add_field_if_not_exists(fc_path, "Dimension", "TEXT", field_length=10)
    add_field_if_not_exists(fc_path, "Chainage", "DOUBLE")
    return fc_path


def build_attribute_field_map(records):
    field_map = {}
    used_names = set()

    for rec in records:
        for key in rec.get("attributes", {}).keys():
            if key in {"Name", "ParentTag", "Dimension", "Chainage"}:
                continue
            safe = clean_name(key.replace(" ", "_").replace("-", "_"), max_length=31)
            if not safe:
                safe = "Value"
            base = safe
            suffix = 1
            while base in used_names:
                base = f"{safe}_{suffix}"
                suffix += 1
            used_names.add(base)
            field_map[key] = base

    return field_map


def write_features(gdb_path: str, records, spatial_reference, limit: int = None) -> None:
    if not records:
        print("No drawable geometry was found in the XML file.")
        return

    grouped = {"POINT": [], "POLYLINE": [], "POLYGON": []}
    for rec in records:
        geometry_type = geometry_type_for(rec["points"])
        grouped.setdefault(geometry_type, []).append(rec)

    for geometry_type, items in grouped.items():
        if not items:
            continue

        dim_group = {"2D": [], "3D": []}
        for rec in items:
            dim_group.setdefault(rec.get("dimension", "2D"), []).append(rec)

        for dimension, dim_items in dim_group.items():
            if not dim_items:
                continue

            fc_name = "Drawable_" + geometry_type.title() + ("_3D" if dimension == "3D" else "_2D")
            fc_path = ensure_feature_class(gdb_path, fc_name, geometry_type, spatial_reference)

            field_map = build_attribute_field_map(dim_items)
            # Create fields and record schema for sanitization
            field_schema = {}
            for src_name, field_name in field_map.items():
                value = dim_items[0].get("attributes", {}).get(src_name)
                if isinstance(value, bool):
                    field_type = "TEXT"
                    field_length = 255
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    field_type = "DOUBLE" if isinstance(value, float) else "LONG"
                    field_length = None
                else:
                    field_type = "TEXT"
                    field_length = 255
                kwargs = {"field_length": field_length} if field_type == "TEXT" and field_length is not None else {}
                add_field_if_not_exists(fc_path, field_name, field_type, **kwargs)
                field_schema[field_name] = (field_type, field_length)

            field_order = [src_name for src_name, _ in sorted(field_map.items(), key=lambda x: x[1])]
            insert_fields = ["SHAPE@", "Name", "ParentTag", "Dimension", "Chainage"] + [field_map[src_name] for src_name in field_order]
            with arcpy.da.InsertCursor(fc_path, insert_fields) as cursor:
                inserted = 0
                for i, rec in enumerate(dim_items):
                    if limit is not None and i >= limit:
                        break

                    try:
                        geometry = build_geometry_for(rec["points"], spatial_reference)
                        chainage = rec.get("chainage")
                        if isinstance(chainage, str):
                            try:
                                chainage = float(chainage)
                            except ValueError:
                                chainage = None
                        values = [geometry, rec["name"], rec["parent_tag"], dimension, chainage]
                        for src_name in field_order:
                            raw = rec.get("attributes", {}).get(src_name)
                            dst_field = field_map[src_name]
                            if raw is None:
                                values.append(None)
                                continue
                            ftype, flen = field_schema.get(dst_field, (None, None))
                            if ftype == "TEXT" and isinstance(raw, str) and flen is not None and len(raw) > flen:
                                print(f"Truncating field '{dst_field}' for record '{rec['name']}' from {len(raw)} to {flen} chars")
                                raw = raw[:flen]
                            values.append(raw)
                        cursor.insertRow(values)
                        inserted += 1
                    except Exception as exc:
                        print(f"Skipping invalid {geometry_type} {dimension} record '{rec['name']}': {exc}")
                        continue

            print(f"Created {inserted} {dimension.lower()} {geometry_type.lower()} records in {fc_name}")


def write_annotations(gdb_path: str, records, spatial_reference) -> None:
    if not records:
        print("No annotation text features were found in the XML file.")
        return

    fc_name = "Drawable_Annotation"
    fc_path = os.path.join(gdb_path, fc_name)
    if arcpy.Exists(fc_path):
        arcpy.management.Delete(fc_path)

    arcpy.management.CreateFeatureclass(
        gdb_path,
        fc_name,
        "POINT",
        spatial_reference=spatial_reference,
    )

    annotation_fields = [
        ("TextString", "TEXT", 255),
        ("TextStyle", "TEXT", 255),
        ("FontSize", "DOUBLE", None),
        ("TextColor", "TEXT", 255),
        ("Rotation", "DOUBLE", None),
        ("Justification", "TEXT", 255),
        ("Offset", "DOUBLE", None),
        ("Raise", "DOUBLE", None),
        ("Name", "TEXT", 255),
        ("ParentTag", "TEXT", 255),
        ("Chainage", "DOUBLE", None),
    ]

    for field_name, field_type, length in annotation_fields:
        kwargs = {}
        if length is not None:
            kwargs["field_length"] = length
        add_field_if_not_exists(fc_path, field_name, field_type, **kwargs)

    with arcpy.da.InsertCursor(
        fc_path,
        ["SHAPE@", "TextString", "TextStyle", "FontSize", "TextColor", "Rotation", "Justification", "Offset", "Raise", "Name", "ParentTag", "Chainage"],
    ) as cursor:
        for rec in records:
            try:
                geometry = arcpy.PointGeometry(arcpy.Point(rec["x"], rec["y"]), spatial_reference)
                chainage = rec.get("chainage")
                if isinstance(chainage, str):
                    try:
                        chainage = float(chainage)
                    except ValueError:
                        chainage = None
                cursor.insertRow([
                    geometry,
                    rec["text"],
                    rec["font_name"],
                    rec["font_size"],
                    rec["text_colour"],
                    rec["rotation"],
                    rec["justification"],
                    rec["offset"],
                    rec["raise"],
                    rec["name"],
                    rec["parent_tag"],
                    chainage,
                ])
            except Exception as exc:
                print(f"Skipping invalid annotation '{rec['name']}': {exc}")
                continue

    print(f"Created {len(records)} annotation records in {fc_name}")


def write_chainage_table(gdb_path: str, records) -> None:
    if not records:
        print("No chainage data found in the XML file.")
        return

    table_name = "Alignment_Chainage"
    table_path = os.path.join(gdb_path, table_name)
    if arcpy.Exists(table_path):
        arcpy.management.Delete(table_path)

    arcpy.management.CreateTable(gdb_path, table_name)
    for field_name, field_type, field_length in [
        ("AlignmentName", "TEXT", 255),
        ("Chainage", "DOUBLE", None),
        ("Breakline", "TEXT", 255),
        ("Colour", "TEXT", 255),
        ("Style", "TEXT", 255),
        ("Dimension", "TEXT", 10),
        ("PointCount", "LONG", None),
    ]:
        kwargs = {}
        if field_length is not None:
            kwargs["field_length"] = field_length
        add_field_if_not_exists(table_path, field_name, field_type, **kwargs)

    with arcpy.da.InsertCursor(table_path, ["AlignmentName", "Chainage", "Breakline", "Colour", "Style", "Dimension", "PointCount"]) as cursor:
        for rec in records:
            try:
                cursor.insertRow([
                    rec["alignment_name"],
                    float(rec["chainage"]),
                    rec["breakline"],
                    rec["colour"],
                    rec["style"],
                    rec["dimension"],
                    len(rec["points"]),
                ])
            except Exception as exc:
                print(f"Skipping invalid chainage record '{rec['alignment_name']}': {exc}")
                continue

    print(f"Created {len(records)} chainage rows in {table_name}")


def write_survey_line_features(gdb_path: str, records, spatial_reference) -> None:
    if not records:
        print("No survey line data found in the XML file.")
        return

    grouped = {"2D": [], "3D": []}
    for rec in records:
        grouped.setdefault(rec.get("dimension", "2D"), []).append(rec)

    for dimension, items in grouped.items():
        if not items:
            continue

        fc_name = "Survey_Line_" + dimension
        fc_path = os.path.join(gdb_path, fc_name)
        if arcpy.Exists(fc_path):
            arcpy.management.Delete(fc_path)

        arcpy.management.CreateFeatureclass(
            gdb_path,
            fc_name,
            "POLYLINE",
            spatial_reference=spatial_reference,
        )

        add_field_if_not_exists(fc_path, "AlignmentName", "TEXT", field_length=255)
        add_field_if_not_exists(fc_path, "Chainage", "DOUBLE")
        add_field_if_not_exists(fc_path, "Breakline", "TEXT", field_length=255)

        with arcpy.da.InsertCursor(fc_path, ["SHAPE@", "AlignmentName", "Chainage", "Breakline"]) as cursor:
            for rec in items:
                try:
                    if not rec["points"]:
                        continue
                    geometry = build_geometry_for(rec["points"], spatial_reference)
                    cursor.insertRow([geometry, rec["alignment_name"], float(rec["chainage"]), rec["breakline"]])
                except Exception as exc:
                    print(f"Skipping invalid survey line '{rec['alignment_name']}': {exc}")
                    continue

        print(f"Created {len(items)} {dimension} survey lines in {fc_name}")


def build_string_super_field_map(records):
    field_map = {}
    used_names = set()

    for rec in records:
        for key in rec["attributes"].keys():
            if key in {"Name", "ParentTag", "Dimension", "Chainage"}:
                continue
            safe = clean_name(key.replace(" ", "_").replace("-", "_"), max_length=31)
            if not safe:
                safe = "Value"
            base = safe
            suffix = 1
            while base in used_names:
                base = f"{safe}_{suffix}"
                suffix += 1
            used_names.add(base)
            field_map[key] = base

    return field_map


def write_string_super_features(gdb_path: str, records, spatial_reference) -> None:
    if not records:
        print("No <string_super> data found in the XML file.")
        return

    grouped = {}
    for rec in records:
        grouped.setdefault(rec["name"], []).append(rec)

    for string_name, items in grouped.items():
        by_dimension = {"2D": [], "3D": []}
        for rec in items:
            by_dimension.setdefault(rec.get("dimension", "2D"), []).append(rec)

        for dimension, dim_items in by_dimension.items():
            if not dim_items:
                continue

            fc_name = "String_Super_" + clean_name(string_name, max_length=40)
            if dimension == "3D":
                fc_name += "_3D"
            else:
                fc_name += "_2D"

            fc_path = os.path.join(gdb_path, fc_name)
            if arcpy.Exists(fc_path):
                arcpy.management.Delete(fc_path)

            geometry_type = geometry_type_for(dim_items[0]["points"])
            arcpy.management.CreateFeatureclass(
                gdb_path,
                fc_name,
                geometry_type,
                spatial_reference=spatial_reference,
            )

            add_field_if_not_exists(fc_path, "Name", "TEXT", field_length=255)
            add_field_if_not_exists(fc_path, "ParentTag", "TEXT", field_length=255)
            add_field_if_not_exists(fc_path, "Dimension", "TEXT", field_length=10)
            add_field_if_not_exists(fc_path, "Chainage", "DOUBLE")

            field_map = build_string_super_field_map(dim_items)
            # Create fields and capture their types/lengths so values can be sanitized before insert
            field_schema = {}
            for src_name, field_name in field_map.items():
                value = dim_items[0]["attributes"].get(src_name)
                if isinstance(value, bool):
                    field_type = "TEXT"
                    field_length = 255
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    if isinstance(value, float):
                        field_type = "DOUBLE"
                    else:
                        field_type = "LONG"
                    field_length = None
                else:
                    field_type = "TEXT"
                    field_length = 255
                kwargs = {"field_length": field_length} if field_type == "TEXT" and field_length is not None else {}
                add_field_if_not_exists(fc_path, field_name, field_type, **kwargs)
                field_schema[field_name] = (field_type, field_length)

            field_order = [src_name for src_name, _ in sorted(field_map.items(), key=lambda x: x[1])]
            insert_fields = ["SHAPE@", "Name", "ParentTag", "Dimension", "Chainage"] + [field_map[src_name] for src_name in field_order]
            with arcpy.da.InsertCursor(fc_path, insert_fields) as cursor:
                for rec in dim_items:
                    try:
                        geometry = build_geometry_for(rec["points"], spatial_reference)
                        values = [geometry, rec["name"], rec["parent_tag"], dimension, rec.get("chainage")]
                        for src_name in field_order:
                            raw = rec["attributes"].get(src_name)
                            dst_field = field_map[src_name]
                            if raw is None:
                                values.append(None)
                                continue

                            ftype, flen = field_schema.get(dst_field, (None, None))
                            if ftype == "TEXT" and isinstance(raw, str) and flen is not None and len(raw) > flen:
                                # Truncate to field length and warn
                                print(f"Truncating field '{dst_field}' for record '{rec['name']}' from {len(raw)} to {flen} chars")
                                raw = raw[:flen]
                            values.append(raw)

                        cursor.insertRow(values)
                    except Exception as exc:
                        print(f"Skipping invalid <string_super> record '{rec['name']}': {exc}")
                        continue

            print(f"Created {len(dim_items)} {dimension} string_super records in {fc_name}")


def write_tin_surfaces(gdb_path: str, tin_records, spatial_reference) -> None:
    if not tin_records:
        print("No full_tin surface data was found in the XML file. Export must include full_tin/points to create an ESRI TIN.")
        return

    tin_folder = os.path.dirname(gdb_path) or "."
    for rec in tin_records:
        clean_name_value = clean_name(rec["name"]).replace("/", "_")
        fc_name = "TIN_MassPoints_" + clean_name_value
        mass_fc_path = os.path.join(gdb_path, fc_name)

        if arcpy.Exists(mass_fc_path):
            arcpy.management.Delete(mass_fc_path)

        arcpy.management.CreateFeatureclass(
            gdb_path,
            fc_name,
            "POINT",
            spatial_reference=spatial_reference,
        )

        add_field_if_not_exists(mass_fc_path, "Z_Value", "DOUBLE")

        with arcpy.da.InsertCursor(mass_fc_path, ["SHAPE@XY", "SHAPE@Z", "Z_Value"]) as cursor:
            for x, y, z in rec["points"]:
                try:
                    cursor.insertRow([(x, y), z, z])
                except Exception as exc:
                    print(f"Skipping invalid TIN point in '{rec['name']}': {exc}")
                    continue

        out_tin = os.path.join(tin_folder, clean_name_value + ".tin")
        if arcpy.Exists(out_tin):
            arcpy.management.Delete(out_tin)

        arcpy.ddd.CreateTin(
            out_tin,
            spatial_reference,
            [[mass_fc_path, "Mass_Points"]],
            "Shape.Z",
        )
        print(f"Created ESRI TIN: {out_tin} from {rec['name']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert 12d XML drawables into File Geodatabase feature classes.")
    parser.add_argument("xml_path", help="Path to the 12d XML file.")
    parser.add_argument("gdb_path", help="Output File Geodatabase path (for example C:/data/output.gdb).")
    parser.add_argument("--spatial-reference", type=int, default=7856, help="ArcGIS spatial reference WKID. Default is 7856 (GDA2020 MGA Zone 56).")
    parser.add_argument("--dry-run", action="store_true", help="Parse the XML and print a summary without creating the File GDB.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of drawable records to write per geometry type.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.xml_path):
        raise FileNotFoundError(f"XML file does not exist: {args.xml_path}")

    root = decode_xml(args.xml_path)
    records = build_feature_records(root)
    annotation_records = build_annotation_records(root)
    tin_records = build_tin_records(root)
    chainage_records = extract_chainage_records(root)
    string_super_records = build_string_super_records(root)
    print(f"Found {len(records)} drawable geometry records in {args.xml_path}")
    print(f"Found {len(annotation_records)} annotation records in {args.xml_path}")
    print(f"Found {len(string_super_records)} <string_super> records in {args.xml_path}")
    print(f"Found {len(tin_records)} full_tin datasets in {args.xml_path}")
    print(f"Found {len(chainage_records)} chainage records in {args.xml_path}")

    counts = {}
    for rec in records:
        geometry_type = geometry_type_for(rec["points"])
        counts[geometry_type] = counts.get(geometry_type, 0) + 1
    print("Geometry summary:", {k: v for k, v in sorted(counts.items())})

    if args.dry_run:
        return 0

    if arcpy is None:
        raise RuntimeError("ArcPy is not available. This script requires ArcGIS Pro / ArcGIS Desktop with arcpy.")

    spatial_reference = arcpy.SpatialReference(args.spatial_reference)
    if not spatial_reference.factoryCode:
        raise ValueError(f"Unsupported spatial reference WKID: {args.spatial_reference}")

    create_file_gdb(args.gdb_path)
    write_features(args.gdb_path, records, spatial_reference, limit=args.limit)
    # write_annotations(args.gdb_path, annotation_records, spatial_reference)
    # write_string_super_features(args.gdb_path, string_super_records, spatial_reference)
    #write_chainage_table(args.gdb_path, chainage_records)
    write_survey_line_features(args.gdb_path, chainage_records, spatial_reference)
    # write_tin_surfaces(args.gdb_path, tin_records, spatial_reference)
    print(f"Finished writing feature classes to {args.gdb_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
