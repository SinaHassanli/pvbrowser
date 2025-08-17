"""Interactive VTK/VTU/VTP/legacy VTK viewer for OpenFOAM-style data using trame.

Goals of this implementation:
 - Reliable field (scalar/vector) selection with automatic magnitude handling for vectors
 - Colormap selection with robust value handling
 - Representation (Points/Wireframe/Surface), opacity, edge visibility controls
 - Camera reset and live updates
 - Verbose debug logging for every state change so mis-wiring can be diagnosed

Design notes:
 - Use VtkRemoteView (server-rendered) for consistency in headless/WLS environments.
 - All UI elements bind to state variables; handlers are registered via state.change("var") decorators.
 - Each handler performs validation, updates underlying VTK pipeline, then calls view.update() so the
     new rendered image is pushed to the client.
 - Colormap application uses a vtkLookupTable; for vector fields we derive a magnitude array if not
     already present (name pattern: <vectorName>_mag).
 - Defensive coding: any user selection that turns out to be a dict (Vuetify can emit dicts depending
     on item config) is normalized to a string before usage.
"""

from __future__ import annotations

import pathlib
import sys
import os
import math
from typing import Dict, List, Tuple

# Configure VTK for headless/server rendering before import
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'

import vtk
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk as vtk_widgets, vuetify

# --------------------------------------------------------------------------------------
# Utility / Data Loading
# --------------------------------------------------------------------------------------


def get_available_fields(data) -> Dict[str, List[str]]:
    fields = {"scalars": [], "vectors": []}
    point_data = data.GetPointData()
    cell_data = data.GetCellData()

    def collect(container, tag: str):
        for i in range(container.GetNumberOfArrays()):
            arr = container.GetArray(i)
            if not arr:
                continue
            name = arr.GetName() or f"array_{i}"
            comps = arr.GetNumberOfComponents()
            if comps == 1:
                fields["scalars"].append(name if tag == "point" else f"{name} (cell)")
            elif comps in (2, 3):  # treat 2 or 3 as vector for resilience
                fields["vectors"].append(name if tag == "point" else f"{name} (cell)")

    collect(point_data, "point")
    collect(cell_data, "cell")
    return fields


def get_colormap_presets():
    return [
        {"text": "Viridis", "value": "viridis"},
        {"text": "Plasma", "value": "plasma"},
        {"text": "Cool to Warm", "value": "coolwarm"},
        {"text": "Rainbow", "value": "rainbow"},
        {"text": "Jet", "value": "jet"},
        {"text": "Hot", "value": "hot"},
        {"text": "Blue to Red", "value": "blue_red"},
        {"text": "Grayscale", "value": "gray"},
    ]


def create_colormap_lut(name: str, num_colors: int = 256):
    name = normalize_choice(name)
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(num_colors)
    name = (name or "gray").lower()
    for i in range(num_colors):
        t = i / (num_colors - 1)
        if name in ("viridis",):
            # Approximation; fine for quick visualization
            r = 0.267 + t * 0.726
            g = 0.005 + t * 0.901
            b = 0.329 - t * 0.185
        elif name in ("plasma",):
            r = 0.05 + t * 0.90
            g = 0.0 + t * 0.80
            b = 0.40 - t * 0.35
        elif name in ("coolwarm", "blue_red"):
            if t < 0.5:
                r = 2 * t
                g = 2 * t
                b = 1.0
            else:
                r = 1.0
                g = 2 * (1 - t)
                b = 2 * (1 - t)
        elif name == "rainbow":
            if t < 0.25:
                r, g, b = 0, 4 * t, 1
            elif t < 0.5:
                r, g, b = 0, 1, 1 - 4 * (t - 0.25)
            elif t < 0.75:
                r, g, b = 4 * (t - 0.5), 1, 0
            else:
                r, g, b = 1, 1 - 4 * (t - 0.75), 0
        elif name == "jet":
            if t < 0.125:
                r, g, b = 0, 0, 0.5 + 4 * t
            elif t < 0.375:
                r, g, b = 0, 4 * (t - 0.125), 1
            elif t < 0.625:
                r, g, b = 4 * (t - 0.375), 1, 1 - 4 * (t - 0.375)
            elif t < 0.875:
                r, g, b = 1, 1 - 4 * (t - 0.625), 0
            else:
                r, g, b = 1 - 4 * (t - 0.875), 0, 0
        elif name == "hot":
            if t < 0.33:
                r, g, b = 3 * t, 0, 0
            elif t < 0.66:
                r, g, b = 1, 3 * (t - 0.33), 0
            else:
                r, g, b = 1, 1, 3 * (t - 0.66)
        else:  # gray fallback
            r = g = b = t
        lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
    lut.Build()
    return lut


def load_data(filename: str):
    if not os.path.isabs(filename):
        filename = os.path.abspath(filename)
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    ext = pathlib.Path(filename).suffix.lower()
    if ext in (".vtp", ".vtu", ".pvtu"):
        if ext == ".vtp":
            reader = vtk.vtkXMLPolyDataReader()
        else:
            reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()
    out = reader.GetOutput()
    if out is None:
        raise RuntimeError("Reader produced no output")
    return out


def build_render_window(data):
    # Force headless rendering for VTK
    import os
    os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'
    
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(data)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(0.8, 0.9, 1.0)
    prop.SetSpecular(0.2)
    prop.SetSpecularPower(10)

    # Points fallback with glyphs
    if data.GetNumberOfCells() == 0 and data.GetNumberOfPoints() > 0:
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.01)
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(data)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.Update()
        mapper.SetInputConnection(glyph.GetOutputPort())

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.13)

    # Don't create scalar bar here - will be created in apply_field_coloring
    scalar_bar = None

    # Create render window with explicit offscreen rendering
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(True)  # Force off-screen rendering
    rw.AddRenderer(renderer)
    rw.SetSize(1800, 1200)  # Higher resolution for clearer text
    renderer.ResetCamera()
    return rw, renderer, actor, mapper, scalar_bar


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def normalize_choice(value):
    if isinstance(value, dict):
        return value.get("value") or value.get("text")
    return value


def ensure_vector_magnitude(data, name: str, is_cell: bool) -> str:
    """If a magnitude array does not exist for vector field 'name', create it; return mag array name."""
    base = name.replace(" (cell)", "")
    mag_name = f"{base}_mag"
    container = data.GetCellData() if is_cell else data.GetPointData()
    if container.HasArray(mag_name):
        return mag_name
    vec = container.GetArray(base)
    if not vec:
        return base  # fallback
    comps = vec.GetNumberOfComponents()
    mag_array = vtk.vtkDoubleArray()
    mag_array.SetName(mag_name)
    mag_array.SetNumberOfComponents(1)
    mag_array.SetNumberOfTuples(vec.GetNumberOfTuples())
    for i in range(vec.GetNumberOfTuples()):
        if comps >= 2:
            v = vec.GetTuple(i)
            m = math.sqrt(sum(c * c for c in v[:comps]))
        else:
            m = vec.GetTuple1(i)
        mag_array.SetTuple1(i, m)
    container.AddArray(mag_array)
    return mag_name


def visualize(filename: str, **start_kwargs):
    print(f"[viewer] Loading file: {filename}")
    data = load_data(filename)
    print(f"[viewer] Points={data.GetNumberOfPoints()} Cells={data.GetNumberOfCells()}")
    fields = get_available_fields(data)
    print(f"[viewer] Scalars={fields['scalars']} Vectors={fields['vectors']}")

    rw, renderer, actor, mapper, scalar_bar = build_render_window(data)
    # Force OSMesa / offscreen if no DISPLAY to avoid X errors
    if not os.environ.get("DISPLAY"):
        try:
            rw.SetOffScreenRendering(1)
        except Exception as e:
            print(f"[offscreen] Unable to enforce offscreen: {e}")
    server = get_server(client_type="vue2")
    state, ctrl = server.state, server.controller

    # Initial state values
    state.setdefault("representation", 2)
    state.setdefault("opacity", 1.0)
    state.setdefault("show_edges", False)
    state.setdefault("use_field_coloring", bool(fields["scalars"]))
    state.setdefault("selected_field", fields["scalars"][0] if fields["scalars"] else "")
    state.setdefault("colormap", "viridis")
    state.setdefault("auto_range", True)
    state.setdefault("field_range_min", 0.0)
    state.setdefault("field_range_max", 1.0)
    state.setdefault("current_range_min", 0.0)
    state.setdefault("current_range_max", 1.0)
    state.setdefault("status", "idle")

    # Will hold the actual view widget once UI is built
    refs = {}

    # -------------------------- Update Helpers ---------------------------------
    def apply_representation():
        rep = int(state.representation)
        p = actor.GetProperty()
        if rep == 0:
            p.SetRepresentationToPoints()
            p.SetPointSize(4)
        elif rep == 1:
            p.SetRepresentationToWireframe()
        else:
            p.SetRepresentationToSurface()
        print(f"[rep] representation={rep}")
        v = refs.get("view")
        if v:
            v.update()

    def apply_opacity():
        op = float(state.opacity)
        actor.GetProperty().SetOpacity(op)
        print(f"[opacity] opacity={op}")
        v = refs.get("view")
        if v:
            v.update()

    def apply_edges():
        show = bool(state.show_edges)
        actor.GetProperty().SetEdgeVisibility(show)
        print(f"[edges] show_edges={show}")
        v = refs.get("view")
        if v:
            v.update()

    def apply_field_coloring():
        use_color = bool(state.use_field_coloring)
        field_name = normalize_choice(state.selected_field)
        print(f"[field] use={use_color} field='{field_name}' colormap={state.colormap}")

        if not use_color or not field_name:
            mapper.ScalarVisibilityOff()
            # Remove any existing scalar bars
            props = renderer.GetViewProps()
            props.InitTraversal()
            for i in range(props.GetNumberOfItems()):
                prop = props.GetNextProp()
                if isinstance(prop, vtk.vtkScalarBarActor):
                    renderer.RemoveViewProp(prop)
            actor.GetProperty().SetColor(0.8, 0.9, 1.0)
            v = refs.get("view")
            if v:
                v.update()
            return
        is_cell = field_name.endswith(" (cell)")
        clean = field_name.replace(" (cell)", "")

        container = data.GetCellData() if is_cell else data.GetPointData()
        arr = container.GetArray(clean)

        if arr and arr.GetNumberOfComponents() in (2, 3):
            mag_name = ensure_vector_magnitude(data, field_name, is_cell)
            arr = container.GetArray(mag_name)
            clean = mag_name

        if not arr:
            print(f"[field] WARNING: array '{clean}' not found")
            mapper.ScalarVisibilityOff()
            # Remove any existing scalar bars
            props = renderer.GetViewProps()
            props.InitTraversal()
            for i in range(props.GetNumberOfItems()):
                prop = props.GetNextProp()
                if isinstance(prop, vtk.vtkScalarBarActor):
                    renderer.RemoveViewProp(prop)
            v = refs.get("view")
            if v:
                v.update()
            return

        if is_cell:
            container.SetActiveScalars(clean)
            mapper.SetScalarModeToUseCellData()
        else:
            container.SetActiveScalars(clean)
            mapper.SetScalarModeToUsePointData()

        if state.auto_range:
            rng = arr.GetRange()
            if (float(state.field_range_min), float(state.field_range_max)) != (float(rng[0]), float(rng[1])):
                state.field_range_min = float(rng[0])
                state.field_range_max = float(rng[1])
                print(f"[range] auto -> {rng}")
        else:
            try:
                vmin = float(state.field_range_min)
                vmax = float(state.field_range_max)
            except Exception:
                base_rng = arr.GetRange()
                vmin, vmax = float(base_rng[0]), float(base_rng[1])
                print(f"[range] parse error; fallback -> {(vmin, vmax)}")
            if vmax < vmin:
                vmin, vmax = vmax, vmin
            rng = (vmin, vmax)
            print(f"[range] manual -> {rng}")

        lut = create_colormap_lut(state.colormap)
        lut.SetRange(rng)
        lut.Build()  # Force rebuild
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(rng)
        mapper.SetColorModeToMapScalars()
        mapper.UseLookupTableScalarRangeOn()
        mapper.ScalarVisibilityOn()
        mapper.SelectColorArray(clean)
        
        # Force scalar bar refresh by removing and re-adding all props
        renderer.RemoveAllViewProps()
        renderer.AddActor(actor)
        
        # Completely recreate scalar bar with forced range display
        scalar_bar = vtk.vtkScalarBarActor()
        
        # Create completely fresh LUT for scalar bar display
        fresh_lut = create_colormap_lut(state.colormap, 256)
        fresh_lut.SetRange(rng[0], rng[1])
        fresh_lut.Build()
        
        # Set the fresh LUT to scalar bar
        scalar_bar.SetLookupTable(fresh_lut)
        scalar_bar.SetTitle(f"{clean}")  # Keep title simple

        # Manually specify tick positions to avoid default 0-250 artifact
        import numpy as _np
        ticks = _np.linspace(rng[0], rng[1], 6)
        # VTK custom labels API (if available) - fallback to SetNumberOfLabels
        try:
            da = vtk.vtkDoubleArray()
            for t in ticks:
                da.InsertNextValue(float(t))
            # Some VTK builds expose SetCustomLabels/UseCustomLabelsOn
            if hasattr(scalar_bar, 'SetCustomLabels'):
                scalar_bar.SetCustomLabels(da)
            if hasattr(scalar_bar, 'UseCustomLabelsOn'):
                scalar_bar.UseCustomLabelsOn()
            # Reduce automatic label count influence
            scalar_bar.SetNumberOfLabels(len(ticks))
        except Exception as e:
            print(f"[scalar_bar] custom label assignment failed: {e}")
            scalar_bar.SetNumberOfLabels(6)
        scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
        scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
        scalar_bar.GetTitleTextProperty().SetFontSize(14)
        scalar_bar.GetLabelTextProperty().SetFontSize(12)
        scalar_bar.SetWidth(0.08)
        scalar_bar.SetHeight(0.4)
        scalar_bar.SetPosition(0.9, 0.3)
        
        # Force rebuild and add to renderer
        scalar_bar.Modified()
        scalar_bar.VisibilityOn()
        renderer.AddViewProp(scalar_bar)
        
        state.current_range_min = float(rng[0])
        state.current_range_max = float(rng[1])
        mapper.Update()
        mapper.Modified()
        
        # Force complete render pipeline refresh
        renderer.Modified()
        rw.Modified()
        rw.Render()  # First render
        rw.Render()  # Second render to ensure scalar bar text updates
        
        # Force view to update with fresh image
        v_local = refs.get("view")
        if v_local:
            v_local.update()
            # Try to force a complete refresh
            if hasattr(v_local, 'reset_camera'):
                v_local.reset_camera()
        
        # Debug prop counts and ACTUAL scalar bar labels being displayed
        num_props = renderer.GetViewProps().GetNumberOfItems()
        sb_lut_range = scalar_bar.GetLookupTable().GetRange() if scalar_bar.GetLookupTable() else None
        
        # Debug prop counts and force scalar bar to recalculate labels
        num_props = renderer.GetViewProps().GetNumberOfItems()
        sb_lut_range = scalar_bar.GetLookupTable().GetRange() if scalar_bar.GetLookupTable() else None
        
        # Force scalar bar to rebuild its labels by rendering once
        scalar_bar.Modified()
        rw.Render()
        
        # Check if the lookup table range matches what we set
        lut_table_range = scalar_bar.GetLookupTable().GetTableRange()
        lut_scalar_range = scalar_bar.GetLookupTable().GetRange()
        
        print(f"[field] applied field='{clean}' range={rng}")
        print(f"[mapper] range={mapper.GetScalarRange()}")
        print(f"[lut] range={lut.GetRange()}")
        print(f"[scalar_bar] lut_range={sb_lut_range}")
        print(f"[scalar_bar] lut_table_range={lut_table_range}")
        print(f"[scalar_bar] lut_scalar_range={lut_scalar_range}")
        print(f"[debug] props={num_props}")
        
        # Check if scalar bar is using the correct lookup table
        if abs(sb_lut_range[0] - rng[0]) > 1e-6 or abs(sb_lut_range[1] - rng[1]) > 1e-6:
            print(f"[ERROR] Scalar bar LUT range {sb_lut_range} does NOT match expected range {rng}")
            # Try to force it again
            scalar_bar.GetLookupTable().SetRange(rng)
            scalar_bar.GetLookupTable().Build()
            scalar_bar.Modified()

    def apply_camera_reset():
        renderer.ResetCamera()
        print("[camera] reset")
        v = refs.get("view")
        if v:
            v.update()

    def apply_manual_range():
        print(f"[manual_range] button clicked - min={state.field_range_min} max={state.field_range_max}")
        apply_field_coloring()

    # -------------------------- State Watchers ---------------------------------
    @state.change("representation")
    def _on_representation(**_):
        apply_representation()

    @state.change("opacity")
    def _on_opacity(**_):
        apply_opacity()

    @state.change("show_edges")
    def _on_edges(**_):
        apply_edges()

    @state.change("use_field_coloring", "selected_field", "colormap", "auto_range")
    def _on_field_update(**_):
        # Ensure selected_field is always stored as a primitive string
        norm = normalize_choice(state.selected_field)
        if isinstance(state.selected_field, dict) or norm != state.selected_field:
            state.selected_field = norm or ""
            print(f"[field] normalized selection -> '{state.selected_field}'")
            # Return so the change triggers this watcher again cleanly without double apply
            return
        apply_field_coloring()

    # Manual range only applies when Apply Range button is clicked

    # Removed direct watcher on min/max to require explicit Apply button (avoids partial input states)

    # Expose controller actions
    ctrl.reset_camera = apply_camera_reset
    ctrl.apply_manual_range = apply_manual_range

    # -------------------------- UI Layout --------------------------------------
    # Provide items via state for reliable serialization (primitive strings)
    state.field_items = ["(None)"] + fields["scalars"] + fields["vectors"]
    preset_defs = get_colormap_presets()
    state.cmap_items = [p["value"] for p in preset_defs]
    print(f"[viewer] field_items={state.field_items}")

    with SinglePageLayout(server) as layout:
        layout.title.set_text("trame VTK Viewer")
        with layout.toolbar:
            vuetify.VBtn("Reset Camera", click=ctrl.reset_camera, small=True, outlined=True)
            vuetify.VSpacer()
            vuetify.VChip(v_model="status", small=True, color="primary", text=True)
        with layout.content:
            with vuetify.VContainer(fluid=True):
                with vuetify.VRow():
                    with vuetify.VCol(cols=3):
                        with vuetify.VCard():
                            vuetify.VCardTitle("Controls")
                            with vuetify.VCardText():
                                vuetify.VSubheader("Field")
                                vuetify.VCheckbox(label="Use Field Coloring", v_model="use_field_coloring")
                                vuetify.VSelect(label="Field", v_model="selected_field", items=("field_items",), dense=True)
                                vuetify.VSelect(label="Colormap", v_model="colormap", items=("cmap_items",), dense=True)
                                vuetify.VCheckbox(label="Auto Range", v_model="auto_range")
                                vuetify.VTextField(label="Min", v_model="field_range_min", type="number", dense=True, disabled=("auto_range",))
                                vuetify.VTextField(label="Max", v_model="field_range_max", type="number", dense=True, disabled=("auto_range",))
                                vuetify.VBtn("Apply Range", click=ctrl.apply_manual_range, small=True, disabled=("auto_range",))
                                vuetify.VList(dense=True, children=[
                                    vuetify.VListItem(title=("Current Min: {{ current_range_min }}")),
                                    vuetify.VListItem(title=("Current Max: {{ current_range_max }}")),
                                ])
                                vuetify.VDivider()
                                vuetify.VSubheader("Appearance")
                                vuetify.VSelect(label="Representation", v_model="representation", items=[
                                    {"text": "Points", "value": 0},
                                    {"text": "Wireframe", "value": 1},
                                    {"text": "Surface", "value": 2},
                                ], dense=True)
                                vuetify.VSlider(label="Opacity", v_model="opacity", min=0, max=1, step=0.05, hide_details=True)
                                vuetify.VCheckbox(label="Show Edges", v_model="show_edges")
                                vuetify.VDivider()
                                vuetify.VSubheader("Info")
                                vuetify.VList(dense=True, children=[
                                    vuetify.VListItem(title=f"Points: {data.GetNumberOfPoints()}")
                                    ,vuetify.VListItem(title=f"Cells: {data.GetNumberOfCells()}")
                                    ,vuetify.VListItem(title=f"Scalars: {len(fields['scalars'])}")
                                    ,vuetify.VListItem(title=f"Vectors: {len(fields['vectors'])}")
                                ])
                    with vuetify.VCol(cols=9):
                        # Use local view with higher resolution for better text clarity
                        refs["view"] = vtk_widgets.VtkLocalView(rw, ref="view")

    # Initial application of settings
    apply_representation()
    apply_opacity()
    apply_edges()
    apply_field_coloring()

    port = start_kwargs.get("port", 8081)
    print(f"[server] Starting on http://localhost:{port}")
    server.start(port=port)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m pvbrowser.viewer <path_to_vtk_or_vtp>")
        sys.exit(1)

    print(f"Starting pvbrowser with file: {sys.argv[1]}")
    visualize(sys.argv[1])
