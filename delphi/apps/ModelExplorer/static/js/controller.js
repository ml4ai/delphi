var docs_to_use = [];
var code_to_use = [];


$(function() {
  $.getJSON("/get_saved_materials",
  {},
  function(data) {
    _.forEach(data["code"], function(code_file) {
      $("#source-file-list").append("<a class=\"list-group-item list-group-item-action\">" + code_file + "</a>");
    });
    _.forEach(data["docs"], function(doc_file) {
      $("#document-list").append("<a class=\"list-group-item list-group-item-action\">" + doc_file + "</a>");
    });

    $("#source-file-list a").on("click", (e) => {
      var filename = $(e.target).text();
      // e.preventDefault();
      // $(e.target).removeClass("active");
      code_to_use.push($(e.target).text());
      console.log(e.target.text);
    });
  });
});

$(function() {
  $("form#doc-upload-form").submit((e) => {
    e.preventDefault();
    var formData = new FormData(this);

    $.ajax({
      url: "/upload_doc",
      type: 'POST',
      data: formData,
      success: function (data) {
          console.log(data);
      },
      error: function(err) { console.log(err); }
      cache: false,
      contentType: false,
      processData: false
    })
  });
  $('#new-doc-modal').modal('hide');
  return false;
  });
});
