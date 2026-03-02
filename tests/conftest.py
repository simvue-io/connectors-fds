import pytest
import simvue
import uuid
import time
import tempfile


@pytest.fixture(scope="session", autouse=True)
def folder_setup():
    # Will be executed before the first test
    folder = "/tests-connectors-%s" % str(uuid.uuid4())
    yield folder
    # Will be executed after the last test
    client = simvue.Client()
    # if client.get_folder(folder):
    #    # Avoid trying to delete folder while one of the runs is still closing
    #    time.sleep(1)
    #    client.delete_folder(folder, remove_runs=True)


@pytest.fixture()
def offline_cache_setup(monkeypatch):
    # Will be executed before test
    cache_dir = tempfile.TemporaryDirectory()
    monkeypatch.setenv("SIMVUE_OFFLINE_DIRECTORY", cache_dir.name)
    yield cache_dir
    # Will be executed after test
    cache_dir.cleanup()

