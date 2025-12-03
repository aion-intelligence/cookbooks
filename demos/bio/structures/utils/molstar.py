import html
import base64
from IPython.display import display, HTML



def preload_molstar():
    """Preload Mol* library at startup"""
    html = """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
    <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
    <div id="molstar-preload-status" style="color: #888; font-size: 12px;">Loading Mol* library...</div>
    <script>
        setTimeout(function() {
            var status = document.getElementById('molstar-preload-status');
            if (typeof rcsbMolstar !== 'undefined') {
                status.innerHTML = '✓ Mol* viewer ready';
                status.style.color = 'green';
            } else {
                status.innerHTML = '✗ Mol* failed to load - visualizations will not work';
                status.style.color = 'red';
            }
        }, 2000);
    </script>
    """
    display(HTML(html))



def show_molstar(cif_content: str, height: int = 600, width: int = 1300, title: str = None) -> None:
    """Display structure using Mol* viewer via iframe srcdoc"""
    
    
    cif_base64 = base64.b64encode(cif_content.encode()).decode()
    
    title_html = f"<h3 style='text-align: center; margin-bottom: 10px;'>{title}</h3>" if title else ""

    # HTML content for srcdoc (needs to be escaped)
    iframe_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.css">
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; }}
            #viewer {{ width: 100%; height: 100vh; }}
            #loading {{ 
                display: flex; 
                align-items: center; 
                justify-content: center; 
                height: 100vh; 
                color: #666;
                font-family: Arial, sans-serif;
            }}
        </style>
    </head>
    <body>
        <div id="loading">Loading Mol* viewer...</div>
        <div id="viewer"></div>
        <script src="https://cdn.jsdelivr.net/npm/@rcsb/rcsb-molstar/build/dist/viewer/rcsb-molstar.js"></script>
        <script>
            var cifBase64 = '{cif_base64}';
            var attempts = 0;
            
            function init() {{
                attempts++;
                if (typeof rcsbMolstar === 'undefined') {{
                    if (attempts < 50) setTimeout(init, 200);
                    else document.getElementById('loading').innerHTML = 'Failed to load Mol*';
                    return;
                }}
                document.getElementById('loading').style.display = 'none';
                var viewer = new rcsbMolstar.Viewer('viewer', {{
                    showImportControls: true,
                    showSessionControls: false,
                    layoutShowSequence: true,
                    layoutShowLog: false,
                    layoutShowLeftPanel: true,
                }});
                var cifData = atob(cifBase64);
                var blob = new Blob([cifData], {{type: 'text/plain'}});
                viewer.loadStructureFromUrl(URL.createObjectURL(blob), 'mmcif', false);
            }}
            init();
        </script>
    </body>
    </html>
    """
    
    # Escape for srcdoc attribute
    escaped_content = html.escape(iframe_content)
    
    wrapper_html = f"""
    {title_html}
    <div style="width: {width}px; margin: 20px auto; border: 2px solid #e0e0e0; border-radius: 10px; overflow: hidden;">
        <iframe 
            srcdoc="{escaped_content}"
            width="100%" 
            height="{height}px" 
            style="border: none;"
            sandbox="allow-scripts"
        ></iframe>
    </div>
    """
    
    display(HTML(wrapper_html))
