import pytest
import simvue
import uuid
import time

@pytest.fixture(scope='session', autouse=True)
def folder_setup():
    # Will be executed before the first test
    folder  = '/tests-connectors-%s' % str(uuid.uuid4())
    yield folder
    # Will be executed after the last test
    client = simvue.Client()
    if client.get_folder(folder):
        # Avoid trying to delete folder while one of the runs is still closing
        time.sleep(1)
        client.delete_folder(folder, remove_runs=True)