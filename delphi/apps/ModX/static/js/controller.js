var SOURCE_DOC = null;
var SOURCE_CODE = null;
var CURR_MODEL = null;

var SOURCE_FILES = "/static/source_model_files/";

var EDITOR = CodeMirror.fromTextArea(document.getElementById("code-viewer-area"), {
    "lineNumbers": "true",
    "viewportMargin": 800,
    "mode": "fortran"
  }
);

// Check for PDF object support
var pdf_support = (PDFObject.supportsPDFs) ? "DOES" : "DOES NOT";
console.log("This browser " + pdf_support + " support inline PDFs.");

$(function() {
  /**
   * Loads the saved document, source code, model, and variable data
   */
  $.getJSON("/get_saved_materials",
  {},
  function(data) {
    // Populate the code/docs/models lists
    _.forEach(data["code"], function(code_file) {
      $("#source-file-list").append("<a class=\"list-group-item list-group-item-action\" role=\"tab\" data-toggle=\"list\">" + code_file + "</a>");
    });
    _.forEach(data["docs"], function(doc_file) {
      $("#document-list").append("<a class=\"list-group-item list-group-item-action\" role=\"tab\" data-toggle=\"list\">" + doc_file + "</a>");
    });
    _.forEach(data["models"], function(model_file) {
      var model_name = model_file.replace(".json", "");
      $("#model-list").append("<a id=model-" + model_name + " class=\"list-group-item list-group-item-action\" role=\"tab\" data-toggle=\"list\">" + model_name + "</a>");
    });

    // Set page update functionality for code/doc/model selection
    $("#source-file-list a").on("click", (e) => {
      e.preventDefault();
      $("#source-file-list a").removeClass("active");
      $(e.target).addClass("active");
      SOURCE_CODE = $(e.target).text();
      $("#model-view-button").addClass("disabled");
      if (SOURCE_CODE !== null && SOURCE_DOC !== null) {
        $("#model-gen-button").removeClass("disabled");
      }
    });
    $("#document-list a").on("click", (e) => {
      e.preventDefault();
      $("#document-list a").removeClass("active");
      $(e.target).addClass("active");
      SOURCE_DOC = $(e.target).text();
      $("#model-view-button").addClass("disabled");
      if (SOURCE_CODE !== null && SOURCE_DOC !== null) {
        $("#model-gen-button").removeClass("disabled");
      }
    });
    $("#model-list a").on("click", (e) => {
      e.preventDefault();
      $("#model-list a").removeClass("active");
      $(e.target).addClass("active");
      CURR_MODEL = $(e.target).text();
      $("#model-view-button").removeClass("disabled");
      $("#source-file-list a").removeClass("active");
      $("#document-list a").removeClass("active");
      SOURCE_DOC = null;
      SOURCE_CODE = null;
      $("#model-gen-button").addClass("disabled");
    });

    // Setup a model for start conditions
    $("#model-SIR-simple").addClass("active");
  });

  /**
   * Uploads a text document to our server from a client for later processing
   * @param  {web-form-data} form A PDF file binary with additional data
   * @return {page-update}      Adds a new element to the document-list
   */
  $("form#doc-upload-form").submit((e) => {
    e.preventDefault();
    var formData = new FormData(this);

    $.ajax({
      url: "/upload_doc",
      type: 'POST',
      data: formData,
      success: function (data) {
        console.log(data);
        $('#new-doc-modal').modal('hide');
        return false;
      },
      error: function(err) { console.log(err); },
      cache: false,
      contentType: false,
      processData: false
    })
  });

  $("#model-gen-button").on("click", (e) => {
    $("#model-load-modal").modal("toggle");
    $("#model-gen-button").addClass("disabled");
    $.ajax({
      url: "/process_text_and_code",
      data: {source_code: SOURCE_CODE, document: SOURCE_DOC},
      type: "POST",
      success: function(data) {
        $("#grounding-accordian").empty();
        _.forEach(data["link_data"], (rows, var_id) => {
          var variable_name_comps = _.split(var_id.replace(/'/g, "")
                                                  .replace("(", "")
                                                  .replace(")", ""), ", ");
          var variable_name = _.join(_.drop(variable_name_comps), "::");
          var variable_uname = _.join(_.drop(variable_name_comps), "__");
          $("#grounding-accordian").append('<div class="card"><div class="card-header" id="heading-' + variable_uname + '"><h4 class="mb-0"><button class="btn btn-link text-dark" data-toggle="collapse" data-target="#collapse-' + variable_uname + '" aria-expanded="true" aria-controls="collapse-' + variable_uname + '">' + variable_name +'</button></h4></div><div id="collapse-' + variable_uname + '" class="collapse" aria-labelledby="heading-' + variable_uname + '" data-parent="#grounding-accordian"><div class="card-body grounding-table"><table class="table table-sm table-hover"><thead><tr><th scope="col">L score</th><th scope="col">Comment</th><th scope="col">V-C score</th><th scope="col">Text-span</th><th scope="col">C-T score</th><th scope="col">Equation symbol</th><th scope="col">T-E score</th></tr></thead><tbody id=body-'+ variable_uname + '></tbody></table></div></div></div>');
          if (rows.length > 0) {
            // Get high scores for all four link measures
            var max_l = _.maxBy(rows, (r) => { return r.link_score; }).link_score;
            var max_vc = _.maxBy(rows, (r) => { return r.vc_score; }).vc_score;
            var max_ct = _.maxBy(rows, (r) => { return r.ct_score; }).ct_score;
            var max_te = _.maxBy(rows, (r) => { return r.te_score; }).te_score;

            // Add row for each entry in table data, bold highest scores
            _.forEach(rows, (r) => {
              var ls = get_grounding_cell(r.link_score, r.link_score == max_l);
              var vc = get_grounding_cell(r.vc_score, r.vc_score == max_vc);
              var ct = get_grounding_cell(r.ct_score, r.ct_score == max_ct);
              var te = get_grounding_cell(r.te_score, r.te_score == max_te);

              var com = get_grounding_cell(r.comm, false);
              var txt = get_grounding_cell(r.txt, false);
              var eqn = get_grounding_cell(r.eqn, false);

              var vals = ls + com + vc + txt + ct + eqn + te;
              $("#body-" + variable_uname).append('<tr>' + vals + '</tr>');
            });
          }
        });

        $("#model-list").empty();
        _.forEach(data["models"], function(model_file) {
          var model_name = model_file.replace(".json", "");
          $("#model-list").append("<a id=model-" + model_name + " class=\"list-group-item list-group-item-action\" role=\"tab\" data-toggle=\"list\">" + model_name + "</a>");
        });
        $("#model-list a").on("click", (e) => {
          e.preventDefault();
          $("#model-list a").removeClass("active");
          $(e.target).addClass("active");
          CURR_MODEL = $(e.target).text();
          $("#model-view-button").removeClass("disabled");
          $("#source-file-list a").removeClass("active");
          $("#document-list a").removeClass("active");
          SOURCE_DOC = null;
          SOURCE_CODE = null;
          $("#model-gen-button").addClass("disabled");
        });
        var pdf_name = get_model_pdf(CURR_MODEL);
        PDFObject.embed(SOURCE_FILES + "docs/" + SOURCE_DOC, "#document-viewer", {height: "700px", pdfOpenParams: { view: 'FitV', page: '2' }});
        $.get(SOURCE_FILES + "code/" + SOURCE_CODE, (code) => {
          EDITOR.getDoc().setValue(code);
        });
        $("#model-load-modal").modal("toggle");
      },
      error: function(data) {
        console.log(err);
        $("#model-load-modal").modal("toggle");
      }
    });
  });

  /**
   * Submits a model to be generated into a GrFN that will be populated to the viewer.
   */
  $("#model-view-button").on("click", (e) => {
    var pdf_name = get_model_pdf(CURR_MODEL);
    PDFObject.embed(SOURCE_FILES + "docs/" + pdf_name, "#document-viewer", {height: "700px", pdfOpenParams: { view: 'FitV', page: '2' }});
    $.get(SOURCE_FILES + "code/" + CURR_MODEL + ".f", (code) => {
      EDITOR.getDoc().setValue(code);
    });

    $.ajax({
      url: '/get_link_table',
      data: { model_json: CURR_MODEL + ".json" },
      type: 'POST',
      success: function(data) {
        $("#grounding-accordian").empty();
        _.forEach(data, (rows, var_id) => {
          var variable_name_comps = _.split(var_id.replace(/'/g, "")
                                                  .replace("(", "")
                                                  .replace(")", ""), ", ");
          var variable_name = _.join(_.drop(variable_name_comps), "::");
          var variable_uname = _.join(_.drop(variable_name_comps), "__");
          $("#grounding-accordian").append('<div class="card"><div class="card-header" id="heading-' + variable_uname + '"><h4 class="mb-0"><button class="btn btn-link text-dark" data-toggle="collapse" data-target="#collapse-' + variable_uname + '" aria-expanded="true" aria-controls="collapse-' + variable_uname + '">' + variable_name +'</button></h4></div><div id="collapse-' + variable_uname + '" class="collapse" aria-labelledby="heading-' + variable_uname + '" data-parent="#grounding-accordian"><div class="card-body grounding-table"><table class="table table-sm table-hover"><thead><tr><th scope="col">L score</th><th scope="col">Comment</th><th scope="col">V-C score</th><th scope="col">Text-span</th><th scope="col">C-T score</th><th scope="col">Equation symbol</th><th scope="col">T-E score</th></tr></thead><tbody id=body-'+ variable_uname + '></tbody></table></div></div></div>');
          if (rows.length > 0) {
            // Get high scores for all four link measures
            var max_l = _.maxBy(rows, (r) => { return r.link_score; }).link_score;
            var max_vc = _.maxBy(rows, (r) => { return r.vc_score; }).vc_score;
            var max_ct = _.maxBy(rows, (r) => { return r.ct_score; }).ct_score;
            var max_te = _.maxBy(rows, (r) => { return r.te_score; }).te_score;

            // Add row for each entry in table data, bold highest scores
            _.forEach(rows, (r) => {
              var ls = get_grounding_cell(r.link_score, r.link_score == max_l);
              var vc = get_grounding_cell(r.vc_score, r.vc_score == max_vc);
              var ct = get_grounding_cell(r.ct_score, r.ct_score == max_ct);
              var te = get_grounding_cell(r.te_score, r.te_score == max_te);

              var com = get_grounding_cell(r.comm, false);
              var txt = get_grounding_cell(r.txt, false);
              var eqn = get_grounding_cell(r.eqn, false);

              var vals = ls + com + vc + txt + ct + eqn + te;
              $("#body-" + variable_uname).append('<tr>' + vals + '</tr>');
            });
          }
        });
      },
      error: (error) => { console.log(error); }
    });
  });

  $("#model-comp-button").on("click", (e) => {
    $.ajax({
        url: '/model_comparison',
        data: {},
        type: 'GET',
        success: function(data) {
          console.log(data);
        },
        error: function(err) {
            console.log(err);
        }
    });
  });

  // Setup a model for start conditions
  CURR_MODEL = "SIR-simple";
  $("#model-view-button").trigger("click");
});

/**
 * Creates table cells for a piece of grounding link data
 * @param  {String}  data   The actual data to be placed in the cell
 * @param  {Boolean} is_max Does the value represent a max value for the col
 * @return {String}         The html string to be appended in the table
 */
function get_grounding_cell(data, is_max) {
  return (is_max) ? "<th scope='row'>"+data+"</th>" : "<td>"+data+"</td>";
}

function get_model_pdf(model_name) {
  switch (model_name) {
    case "SIR-simple":
      return "ideal_sir_model.pdf";
      break;
    case "petasce":
      return "2005-THE ASCE STANDARDIZED REFERENCE EVAPOTRANSPIRATION EQUATION.pdf";
      break;
    case "SIR-Gillespie":
      return "sir_gillespie_model.pdf";
      break;
    default:
      return "ideal_sir_model.pdf";
      break;
  }
}
