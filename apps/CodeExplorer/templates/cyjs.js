        var computational_graph = cytoscape({
          container: document.getElementById('computational_graph'),
            elements: {{ scopeTree_elementsJSON | safe }},
            style: [
                    { selector: 'node',
                      style: { 
                          'label': 'data(label)',
                          'shape': 'data(shape)',
                          'background-color': 'white',
                          'border-color': 'data(color)',
                          'border-width': '3pt',
                          'font-family': 'Menlo, PT Sans, sans-serif',
                          'width': 'label',
                          'height': 'data(height)',
                          'text-valign': 'data(textValign)',
                          'padding': 'data(padding)',
                      } 
                    }, { 
                      selector: 'edge',
                      style: { 
                        'curve-style' : 'bezier',
                        'target-arrow-shape': 'triangle',
                      } 
                    }, { 
                      selector: '.selectedNode',
                      style: { 
                        'background-color': '#d3d3d3',
                      } 
                    }
                ],
            layout: { name: 'dagre' , rankDir: 'TB'},
            maxZoom : 2,
            minZoom : 0.1,
            selectionType: 'additive'
        });

        var api = computational_graph.expandCollapse({
            fisheye: false, undoable: false
        });

        var makeTippy = function(node){
            return tippy(node.popperRef(), {
                html: (function(){
                    var div = document.createElement('div');
                    div.innerHTML = node.data('tooltip');
                    return div;
                })(),
                trigger: 'manual',
                placement: 'bottom',
                arrow: true,
                hideOnClick: 'toggle',
                multiple: true,
                sticky: true,
                interactive: true,
                theme: 'light',
            }).tooltips[0];
        };
        computational_graph.nodes().forEach(function(ele){
            ele.scratch()._tippy = makeTippy(ele);
        });

        computational_graph.nodes().on("expandcollapse.afterexpand", function(event) {
          var node = event.target;
          node.deselect();
          node.toggleClass('selectedNode');
        })

        computational_graph.nodes().on("expandcollapse.aftercollapse", function(event) {
          var node = event.target;
          node.deselect();
          node.toggleClass('selectedNode');
        })

        computational_graph.on('tap', 'node', function(evt){
          var node = evt.target;
            if (!node.selected()){
              if (!node.hasClass('cy-expand-collapse-collapsed-node') && !node.isParent()) {
                node.scratch()._tippy.show();
                MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
              }
            }
            else {
              node.scratch()._tippy.hide();
            }
            node.toggleClass('selectedNode');
        });

        var causal_analysis_graph = cytoscape({
        container: document.getElementById('causal_analysis_graph'),
          elements: {{ program_analysis_graph_elementsJSON | safe }},
          style: [
                  { selector: 'node',
                    style: { 
                        'label': 'data(label)',
                        'shape': 'data(shape)',
                        'background-color': 'white',
                        'border-color': 'data(color)',
                        'border-width': '3pt',
                        'font-family': 'PT Sans, sans-serif',
                        'width':'data(width)',
                        'height':'data(height)',
                        'text-valign': 'data(textValign)',
                        'padding': 6,
                    } 
                  }, { 
                    selector: 'edge',
                    style: { 
                      'curve-style' : 'bezier',
                      'control-point-step-size': '70px',
                      'target-arrow-shape': 'triangle',
                    } 
                  }, { 
                    selector: '.selectedNode',
                    style: { 
                      'background-color': '#d3d3d3',
                    } 
                  }
              ],
          layout: { name: 'dagre' , rankDir: 'TB'},
          maxZoom : 1,
          minZoom : 0.1,
          selectionType: 'single'
        });
       causal_analysis_graph.pan({x:70,y:70});
