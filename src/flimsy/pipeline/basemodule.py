# Modules are data-centric, atomic units designed to accomplish one task (e.g.,
# analyseCameraWithLabjack or analyzeCameraWithoutLabjack). Modules specify 
# inputs, outputs, and a name in module decorators. Modules are automatically
# registered by these decorators with a central repository, which is used to 
# construct and execute pipelines (see flimsy.pipeline.registry and 
# flimsy.pipeline.pipeline for more information). ## TODO -- change pipeline to runner or something

# Modules automatically perform basic validation of their output. For example,
# modules will flag shape and datatype mismatches (e.g., 0 length arrays and 
# arrays with type object or string when int/float was expected). They will
# additionally check for nan/infs. Results of the validation are shown in the 
# pipeline summary at the end of execution (in addition to major errors such 
# as crashes)

import logging

from flimsy.pipeline.registry import module, requires, produces, param

logger = logging.getLogger(__name__)


def validate_output(module_name, output):
    pass

def should_overwrite():
    pass