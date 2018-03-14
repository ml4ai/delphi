cyStyle = [
          { selector: "node",
            style: { 
              "label": "data(id)",
                "background-color": "white",
                "border-color": "maroon",
                "border-width": "1px",
                "font-family": "Arno Pro, Arial",
                "text-halign": "center",
                "text-valign": "center",
                "padding": 10,
                "width":"label",
                "shape": "roundrectangle",
                "text-max-width": 80,
                "text-wrap": true
            } 
          }, { 
            selector: "edge",
            style: { 
              "curve-style" : "bezier",
                "line-color": "data(linecolor)",
                "target-arrow-shape": "data(targetArrowShape)",
                "target-arrow-color" : "data(linecolor)",
                "line-style" : "data(linestyle)",
                "width": "1"
            } 
          }
        ]
