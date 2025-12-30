import logging

logger = logging.getLogger(__name__)

from flimsy.pipeline.registry import module

@module(name='TestModule', requires=['param1', 'param2'], produces=['status_string'])
def run_test_module(param1, param2):
   logger.debug('Module ran successfully')