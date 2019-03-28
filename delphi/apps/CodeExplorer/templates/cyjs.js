// Set the style for the graphs
var cyjs_style = [
  { 
    selector: 'node',
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
]

// Set the layout for the graphs
var cyjs_layout = { 
  name: 'dagre',
  rankDir: 'TB',
  nodeDimensionsIncludeLabels: true,
}

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

// This function creates the cytoscape graph objects
var make_cyjs = function(graph_name, elementsJSON){
  var G = cytoscape({
    container: document.getElementById(graph_name),
      elements: elementsJSON,
      style: cyjs_style,
      layout: cyjs_layout,
      maxZoom : 2,
      minZoom : 0.1,
      selectionType: 'additive'
  });
  G.nodes().forEach(function(ele){
      ele.scratch()._tippy = makeTippy(ele);
  });
  G.on('tap', 'node', function(evt){
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
  return G
}

// ====================================
// Computational Graph
// ====================================

var computational_graph = make_cyjs(
  'computational_graph',
  {{ scopeTree_elementsJSON | safe }}
); 
var api = computational_graph.expandCollapse({
    fisheye: false, undoable: false
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


// ====================================
// Causal Analysis Graph
// ====================================

var causal_analysis_graph = make_cyjs(
  'causal_analysis_graph',
  {{ program_analysis_graph_elementsJSON | safe }}
);
causal_analysis_graph.pan({x:70,y:70});

// ====================================
// Forward Influence Blanket
// ====================================
var forward_influence_blanket = make_cyjs(
  'fib',
  {{ fib_elementsJSON | safe }}
);

