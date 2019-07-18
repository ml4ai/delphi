#!/usr/bin/python
#
#  Copyright 2017 Otto Seiskari
#  Licensed under the Apache License, Version 2.0.
#  See http://www.apache.org/licenses/LICENSE-2.0 for the full text.
#
#  This file is based on
#  https://github.com/swagger-api/swagger-ui/blob/4f1772f6544699bc748299bd65f7ae2112777abc/dist/index.html
#  (Copyright 2017 SmartBear Software, Licensed under Apache 2.0)
#
"""
Usage:
    
    python swagger-yaml-to-html.py < /path/to/api.yaml > doc.html

"""

import io
import sys
import json
from ruamel.yaml import YAML

TEMPLATE = """
  <link href="https://fonts.googleapis.com/css?family=Open+Sans:400,700|Source+Code+Pro:300,600|Titillium+Web:400,600,700" rel="stylesheet">
  <link rel="stylesheet" type="text/css" href="_static/swagger-ui/swagger-ui.css" >
  <style>
    html
    {
      box-sizing: border-box;
      overflow: -moz-scrollbars-vertical;
      overflow-y: scroll;
    }
    *,
    *:before,
    *:after
    {
      box-sizing: inherit;
    }

    body {
      margin:0;
      background: #fafafa;
    }
  </style>

<div id="swagger-ui"></div>
<script src="_static/swagger-ui/swagger-ui-bundle.js"> </script>
<script src="_static/swagger-ui/swagger-ui-standalone-preset.js"> </script>
<script>
    window.onload = function() {

    // Build a system
    var spec = %s;
    const ui = SwaggerUIBundle({
        spec: spec,
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
            SwaggerUIBundle.presets.apis,
            SwaggerUIStandalonePreset
        ],
        plugins: [
            SwaggerUIBundle.plugins.DownloadUrl
        ],
        layout: "StandaloneLayout"
    })

    window.ui = ui
    }
</script>
"""

yaml = YAML()

with open(sys.argv[1], "r") as f:
    data = yaml.load(f)

with open(sys.argv[2], "w") as f:
    f.write(TEMPLATE % format(json.dumps(data)))
