GrFN Json
=========

.. json:object:: GrFN

   What Github's API thinks a user looks like.

   :property str date_created: Date created
   :property `[Identifier]` identifiers: Identifiers

.. json:object:: Identifier

   Each identifier within a GrFN specification will have a single
   `Identifier` declaration. An identifier will be declared in the GrFN
   spec JSON by the following attribute-value list:

   :property str base_name:
   :property str scope:
   :property str namespace:
   :property str source_references:
   :property `[GroundingMetaData]` grounding:


.. json:object:: GroundingMetadata

   :property str source:
   :property str type:
   :property str value:
