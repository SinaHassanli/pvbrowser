"""Alternative viewer using vtkColorTransferFunction to fix scalar bar range issues.
Run: python3 -m pvbrowser.viewer_ctf /path/to/data.vtk
"""
import sys, os, math
import vtk
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vtk as vtk_widgets, vuetify3 as vuetify

# ---------------- Colormap definitions (static to avoid extra deps) -----------------
VIRIDIS_STOPS = [
    (0.0, 0.267004, 0.004874, 0.329415),
    (0.13, 0.282623, 0.140926, 0.457517),
    (0.25, 0.253935, 0.265254, 0.529983),
    (0.38, 0.206756, 0.371758, 0.553117),
    (0.50, 0.163625, 0.471133, 0.558148),
    (0.63, 0.127568, 0.566949, 0.550556),
    (0.75, 0.134692, 0.658636, 0.517649),
    (0.88, 0.266941, 0.748751, 0.440573),
    (1.0, 0.477504, 0.821444, 0.318195),
]

PLASMA_STOPS = [
    (0.0, 0.050383, 0.029803, 0.527975),
    (0.13, 0.239374, 0.030909, 0.617407),
    (0.25, 0.433756, 0.030058, 0.658341),
    (0.38, 0.620116, 0.08436, 0.636902),
    (0.50, 0.758422, 0.214607, 0.570856),
    (0.63, 0.854645, 0.34334, 0.470427),
    (0.75, 0.916242, 0.471899, 0.353413),
    (0.88, 0.953099, 0.600837, 0.230311),
    (1.0, 0.940015, 0.975158, 0.131326),
]

JET_STOPS = [
    (0.0, 0, 0, 0.5), (0.35, 0, 0, 1.0), (0.5, 0, 1.0, 1.0), (0.65, 1.0, 1.0, 0), (0.85, 1.0, 0, 0), (1.0, 0.5, 0, 0)
]

COLORMAPS = {
    "viridis": VIRIDIS_STOPS,
    "plasma": PLASMA_STOPS,
    "jet": JET_STOPS,
}

# -------------------------- Data loading ------------------------------------------

def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    r = None
    if ext == ".vtp":
        r = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtu":
        r = vtk.vtkXMLUnstructuredGridReader()
    elif ext == ".vtk":
        # try polydata then unstructured then legacy image/structured
        r = vtk.vtkDataSetReader()
    else:
        raise RuntimeError(f"Unsupported extension {ext}")
    r.SetFileName(path)
    r.Update()
    data = r.GetOutput()
    if data is None or data.GetNumberOfPoints() == 0:
        raise RuntimeError("Failed to load data")
    return data

# --------------------- Color transfer + scalar bar builder -------------------------

def build_ctf(name, vmin, vmax):
    stops = COLORMAPS.get(name, VIRIDIS_STOPS)
    ctf = vtk.vtkColorTransferFunction()
    ctf.RemoveAllPoints()
    for p, r, g, b in stops:
        ctf.AddRGBPoint(vmin + p * (vmax - vmin), r, g, b)
    return ctf

# --------------------- Field discovery --------------------------------------------

def list_fields(data):
    scalars = []
    vectors = []
    for container in (data.GetPointData(), data.GetCellData()):
        for i in range(container.GetNumberOfArrays()):
            arr = container.GetArray(i)
            if not arr: continue
            name = arr.GetName() or f"array_{i}"
            if arr.GetNumberOfComponents() == 1:
                if name not in scalars: scalars.append(name)
            elif arr.GetNumberOfComponents() >= 2:
                if name not in vectors: vectors.append(name)
    return scalars, vectors

# ---------------------- Viewer logic ----------------------------------------------

def main(path):
    data = load_any(path)
    scalars, vectors = list_fields(data)
    file_name = os.path.basename(path)

    # Generate magnitude arrays for vectors
    for vname in vectors:
        arr = data.GetPointData().GetArray(vname) or data.GetCellData().GetArray(vname)
        if arr and data.GetPointData().GetArray(vname):
            mag = vtk.vtkDoubleArray(); mag.SetName(f"{vname}_mag")
            n = arr.GetNumberOfTuples(); mag.SetNumberOfTuples(n)
            for i in range(n):
                comp = arr.GetTuple(i)
                mag.SetValue(i, math.sqrt(sum(c*c for c in comp)))
            data.GetPointData().AddArray(mag)
            scalars.append(f"{vname}_mag")

    # ---------------- Single view setup ----------------
    mapper = vtk.vtkDataSetMapper(); mapper.SetInputData(data)
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    renderer = vtk.vtkRenderer(); renderer.AddActor(actor); renderer.SetBackground(0.1,0.1,0.13)
    rw = vtk.vtkRenderWindow(); rw.OffScreenRenderingOn(); rw.AddRenderer(renderer); rw.SetSize(1400,900)
    scalar_bar = vtk.vtkScalarBarActor(); scalar_bar.SetWidth(0.1); scalar_bar.SetHeight(0.8); scalar_bar.SetPosition(0.9,0.1)
    scalar_bar.GetTitleTextProperty().SetColor(1,1,1); scalar_bar.GetLabelTextProperty().SetColor(1,1,1)
    renderer.AddViewProp(scalar_bar)
    # View direction label
    view_text = vtk.vtkTextActor()
    view_text.GetTextProperty().SetFontSize(18)
    view_text.GetTextProperty().SetColor(1,1,1)
    view_text.SetInput("+Z")
    renderer.AddActor2D(view_text)
    # Camera helpers
    b = data.GetBounds(); cx=(b[0]+b[1])/2; cy=(b[2]+b[3])/2; cz=(b[4]+b[5])/2
    dx=b[1]-b[0]; dy=b[3]-b[2]; dz=b[5]-b[4]
    diag=max(1e-6, math.sqrt(dx*dx+dy*dy+dz*dz))
    base_dist=1.8*diag
    view_ref = []  # reference to single vtk view widget (defined early for camera helper)
    def orient_camera(mode="ISO"):
        print(f"[camera] orient_camera called with mode={mode}")
        cam = renderer.GetActiveCamera()
        # Start from a reset to ensure consistent zoom/framing
        renderer.ResetCamera()
        if mode == "+X":
            cam.SetPosition(cx+base_dist, cy, cz); cam.SetViewUp(0,0,1)
        elif mode == "+Y":
            cam.SetPosition(cx, cy+base_dist, cz); cam.SetViewUp(0,0,1)
        elif mode == "+Z":
            cam.SetPosition(cx, cy, cz+base_dist); cam.SetViewUp(0,1,0)
        else:  # ISO
            cam.SetPosition(cx+base_dist, cy+base_dist, cz+base_dist); cam.SetViewUp(0,0,1)
        cam.SetFocalPoint(cx,cy,cz)
        renderer.ResetCameraClippingRange()
        # Text overlay
        try:
            view_text.SetInput(mode)
        except Exception as e:
            print(f"[camera] text update error: {e}")
        try:
            rw.Render()
        except Exception as e:
            print(f"[camera] render error: {e}")
        # Force view update using ctrl.view_update
        try:
            ctrl.view_update()
            print(f"[camera] ctrl.view_update called successfully")
        except Exception as e:
            print(f"[camera] ctrl.view_update error: {e}")
        # Push camera to client (ensures orientation reflects in frontend)
        try:
            ctrl.view_push_camera()
            print("[camera] push_camera sent")
        except Exception as e:
            print(f"[camera] push_camera error: {e}")
        try:
            print(f"[camera] mode={mode} pos={cam.GetPosition()} fp={cam.GetFocalPoint()} up={cam.GetViewUp()}")
        except:
            pass
    # Will call orient_camera after widget creation so updates succeed

    # Create server early so widgets get proper context
    # Explicitly request vue3 client so widget set matches (important!)
    server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller
    state.field = scalars[0] if scalars else ""
    state.colormap = "viridis"
    state.auto = True
    state.rmin = 0.0
    state.rmax = 1.0
    state.current_range_label = ""
    state.filename = file_name

    # view_ref already created above

    def compute_range(name):
        # Find array by name searching point then cell
        for container in (data.GetPointData(), data.GetCellData()):
            arr = container.GetArray(name)
            if arr:
                r = arr.GetRange()
                return float(r[0]), float(r[1])
        return 0.0,1.0

    _applying = {"flag": False}

    def apply():
        if _applying["flag"]:
            return
        _applying["flag"] = True
        try:
            fname = state.field
            if not fname:
                mapper.ScalarVisibilityOff(); rw.Render()
                for w in view_ref:
                    try: w.update()
                    except: pass
                return
            # Update range
            if state.auto:
                vmin, vmax = compute_range(fname)
                state.rmin, state.rmax = vmin, vmax
            else:
                vmin, vmax = float(state.rmin), float(state.rmax)
                if vmax < vmin:
                    vmin, vmax = vmax, vmin
            # Continuous color transfer function mapping full range
            ctf = build_ctf(state.colormap, vmin, vmax)
            mapper.SetLookupTable(ctf)
            mapper.SetScalarRange(vmin, vmax)
            mapper.SetColorModeToMapScalars(); mapper.UseLookupTableScalarRangeOn(); mapper.ScalarVisibilityOn()
            if data.GetPointData().GetArray(fname):
                mapper.SetScalarModeToUsePointFieldData(); mapper.SelectColorArray(fname)
            else:
                mapper.SetScalarModeToUseCellFieldData(); mapper.SelectColorArray(fname)
            scalar_bar.SetLookupTable(ctf)
            scalar_bar.SetTitle(f"{fname} [{vmin:.3g}, {vmax:.3g}]")
            scalar_bar.SetNumberOfLabels(6)
            scalar_bar.Modified()
            # Do not reset orientation during apply so user-chosen view persists
            try:
                rw.Render()
            except Exception as e:
                print(f"[render] ERROR: {e}")
            for w in view_ref:
                try:
                    w.update()
                except Exception as e:
                    print(f"[view] update error: {e}")
            bounds = actor.GetBounds() if actor else None
            print(f"[debug] bounds={bounds} point_count={data.GetNumberOfPoints()} cell_count={data.GetNumberOfCells()}")
            state.current_range_label = f"{vmin:.4g} to {vmax:.4g}"
            print(f"[apply] field={fname} range=({vmin},{vmax}) ctf_range={ctf.GetRange()} mapper_range={mapper.GetScalarRange()}")
        finally:
            _applying["flag"] = False

    @state.change("field","colormap","auto")
    def _update(**_):
        apply()

    def manual():
        state.auto = False
        apply()
    ctrl.apply_manual = manual
    def reset_cam():
        orient_camera("ISO")
    ctrl.reset_camera = reset_cam
    def _v_iso(): orient_camera("ISO")
    def _v_px(): orient_camera("+X")
    def _v_py(): orient_camera("+Y")
    def _v_pz(): orient_camera("+Z")
    ctrl.view_iso = _v_iso
    ctrl.view_pos_x = _v_px
    ctrl.view_pos_y = _v_py
    ctrl.view_pos_z = _v_pz

    with SinglePageLayout(server) as layout:
        layout.title.set_text("Arup Paraview Viewer")
        with layout.toolbar:
            vuetify.VSpacer()
            vuetify.VBtn("ISO", click=ctrl.view_iso, density="comfortable", variant="text")
            vuetify.VBtn("+X", click=ctrl.view_pos_x, density="comfortable", variant="text")
            vuetify.VBtn("+Y", click=ctrl.view_pos_y, density="comfortable", variant="text")
            vuetify.VBtn("+Z", click=ctrl.view_pos_z, density="comfortable", variant="text")
            vuetify.VChip(v_model=("filename", file_name), color="grey-darken-2", variant="outlined", size="small")
        with layout.content:
            # Two-column layout: left controls, right render view
            with vuetify.VRow(no_gutters=True, classes="ma-0 pa-0 fill-height", style="height:100%;"):
                with vuetify.VCol(cols=3, classes="pa-2 d-flex flex-column", style="max-width:320px; overflow-y:auto; background:#222; color:#ddd; border-right:1px solid #333;"):
                    vuetify.VListSubheader("Field Coloring")
                    vuetify.VSelect(label="Field", items=("field_items", scalars), v_model=("field", state.field), density="comfortable", hide_details=True)
                    vuetify.VSelect(label="Colormap", items=("cmaps", list(COLORMAPS.keys())), v_model=("colormap","viridis"), density="comfortable", hide_details=True)
                    vuetify.VCheckbox(v_model=("auto", True), label="Auto Range", density="compact", hide_details=True)
                    with vuetify.VRow(classes="mt-0"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("rmin", 0.0), label="Min", type="number", density="compact", hide_details=True)
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("rmax", 1.0), label="Max", type="number", density="compact", hide_details=True)
                    with vuetify.VRow(classes="mt-1"):
                        with vuetify.VCol(cols=6):
                            vuetify.VBtn("Apply", block=True, variant="outlined", color="primary", click=ctrl.apply_manual, density="compact")
                        with vuetify.VCol(cols=6):
                            vuetify.VBtn("Reset", block=True, variant="outlined", color="secondary", click=ctrl.reset_camera, density="compact")
                    # Removed Current Range and Help sections per user request
                with vuetify.VCol(cols=9, classes="pa-0 d-flex flex-column", style="height:100%; background:#000; position:relative;"):
                    v_inst = vtk_widgets.VtkLocalView(rw, ref="view_main", ctx_name="view", interactive_ratio=1, style="flex:1 1 auto; width:100%; height:100%; cursor:grab;")
                    view_ref.append(v_inst)
                    # Store view update method in controller
                    ctrl.view_update = v_inst.update
                    ctrl.view_reset_camera = v_inst.reset_camera
                    ctrl.view_push_camera = v_inst.push_camera
                    orient_camera("+Z")

    def _client_ready(**_):
        print("[server] client ready -> forcing apply + render")
        apply()
    try:
        server.add_connection_ready_handler(_client_ready)
    except Exception:
        pass

    state.field_items = scalars
    state.cmaps = list(COLORMAPS.keys())
    apply()
    server.start()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m pvbrowser.viewer_ctf <data file>"); sys.exit(1)
    main(sys.argv[1])
