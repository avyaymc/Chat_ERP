import webbrowser
from utils import update_type
def open_graph(s):
    update_type('type.txt',s)
    webbrowser.open_new_tab("http://localhost:8888/dashboard")
    return f"A {s} graph has been generated and should now be visible in another window."