import json
from pathlib import Path
from unittest.mock import patch

from spherepackgen.cli import main
from spherepackgen.gui.paths import remembered_runs, remember_run, workspace_root


def test_workspace_uses_launch_location_or_explicit_override(tmp_path, monkeypatch):
    monkeypatch.delenv("SPHEREPACKGEN_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert workspace_root() == tmp_path.resolve()
    override = tmp_path / "other"
    monkeypatch.setenv("SPHEREPACKGEN_WORKSPACE", str(override))
    assert workspace_root() == override.resolve()


def test_gui_launch_uses_packaged_app_and_user_workspace(tmp_path):
    from spherepackgen.gui.launcher import launch_gui
    with patch("spherepackgen.gui.launcher.subprocess.call", return_value=0) as call:
        assert launch_gui(tmp_path / "workspace", 9123, True) == 0
    args, kwargs = call.call_args
    command = args[0]
    assert Path(command[4]).is_file()
    assert kwargs["cwd"] == (tmp_path / "workspace").resolve()
    assert kwargs["env"]["SPHEREPACKGEN_WORKSPACE"] == str(kwargs["cwd"])
    assert command[command.index("--server.port") + 1] == "9123"
    assert command[command.index("--server.address") + 1] == "127.0.0.1"


def test_packaged_examples_can_be_exported_without_checkout(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["examples", "--output", "configs"]) == 0
    files = json.loads(capsys.readouterr().out)["files"]
    assert "fcc.yaml" in files
    before = (tmp_path / "configs/fcc.yaml").read_bytes()
    assert main(["examples", "--output", "configs"]) == 1
    assert "already exists" in capsys.readouterr().err
    assert (tmp_path / "configs/fcc.yaml").read_bytes() == before


def test_external_result_directory_is_remembered_after_restart(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    result = tmp_path / "external-results"
    result.mkdir()
    (result / "metadata.json").write_text("{}")
    remember_run(workspace, result)
    remember_run(workspace, result)
    assert remembered_runs(workspace) == [result.resolve()]
    from spherepackgen.gui import streamlit_app
    monkeypatch.setattr(streamlit_app, "ROOT", workspace)
    assert streamlit_app._result_history_options() == [str(result.resolve())]
