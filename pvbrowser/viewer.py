import pathlib

import vtk
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk as vtk_widgets


def load_data(filename: str):
    """Return a vtkDataSet loaded from *filename*."""
    ext = pathlib.Path(filename).suffix.lower()
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    else:
        # Fall back to legacy reader
        reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def build_render_window(data):
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(data)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.ResetCamera()
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    return render_window


def visualize(filename: str, **start_kwargs):
    """Start a trame server visualising *filename*."""
    data = load_data(filename)
    render_window = build_render_window(data)
    server = get_server(client_type="vue2")
    with SinglePageLayout(server) as layout:
        layout.title.set_text("VTK Browser Viewer")
        with layout.content:
            view = vtk_widgets.VtkLocalView(render_window)
    view.update()
    server.start(**start_kwargs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pvbrowser.viewer <path_to_vtk_or_vtp>")
        sys.exit(1)

    visualize(sys.argv[1])
