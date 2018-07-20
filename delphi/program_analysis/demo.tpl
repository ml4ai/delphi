{% extends 'full.tpl'%}
{% block output_group %}
<div class="output_wrapper" 
style="max-height: 400px; overflow: scroll">
<div class="output" >
{{ super() }}
</div>
</div>
{% endblock output_group %}
