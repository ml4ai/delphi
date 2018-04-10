var cy = cytoscape({
  container: document.getElementById('cy'),
  elements: {{ state.elementsJSONforJinja | safe }},
  style: [
    { selector: 'node',
      style: { 
        'label': 'data(id)',
          'background-color': 'white',
          'border-color': 'maroon',
          'border-width': '3pt',
          'font-family': 'Arno Pro, Arial',
          'width':'30',
      } 
    }, { 
      selector: 'edge',
      style: { 
        'curve-style' : 'bezier',
          'line-color': 'maroon',
          'target-arrow-shape': 'circle',
          'target-arrow-color' : 'maroon',
          'line-style' : 'data(linestyle)'
      } 
    },
  ],
  layout: { name: 'cola' },                
  maxZoom : 10,
  minZoom : 0.1,
});
var makeTippy = function(node, text){
  return tippy( node.popperRef(), {
    html: (function(){
      var div = document.createElement('div');
      div.innerHTML = text;
      return div;
    })(),
    trigger: 'manual',
    arrow: true,
    placement: 'top',
    hideOnClick: false,
    interactive: true,
    multiple: true,
  }).tooltips[0];
};

cy.elements().forEach(function(ele){
  ele.data()['tip'] = makeTippy(ele, ele.id());
});


cy.on('tap', function(evt){
  var ele = evt.target;
  if (ele.data()['tip']['state']['visible']){
    ele.data()['tip'].hide();
  } else {
    ele.data()['tip'].show();
  }
});
