$SCRIPT_ROOT = {{ request.script_root|tojson|safe }};

var editor_for_source_code = CodeMirror.fromTextArea(
  document.getElementById('flask-codemirror-source_code'), {
    "lineNumbers": "true",
    "viewportMargin": 800,
    "mode": "fortran"
  }
);

function getCode(programName) {
  fetch('static/example_programs/'+programName+'.f')
    .then(
      function(response) {
        if (response.status !== 200) {
          console.log('Looks like there was a problem. Status Code: ' +
            response.status);
          return;
        }

        // Examine the text in the response
        response.text().then(function(data) {
          editor_for_source_code.getDoc().setValue(data);
        });
      }
    )
    .catch(function(err) {
      console.log('Fetch Error :-S', err);
    });
}

{% for program in ('petPT', 'petASCE', 'cropYield', 'SIR-simple') %}
  $("#{{ program }}").click(function(){
    getCode("{{ program }}");
  });
{% endfor %}
