"""环境组合事务必须兼容真实 v1 外观事务，不能只验证附加属性齐全的 fake。"""
from types import SimpleNamespace
from ue import run_task


def test_combined_callback_accepts_actual_v1_transaction(tmp_path, monkeypatch):
    events = []
    appearance = run_task._AppearanceTransaction(
        None, tmp_path / 'rollback.json', tmp_path / 'application.json',
        tmp_path / 'public.json',
    )
    def restore_appearance(status, result=None):
        events.append(('appearance', status))
        appearance.restore_result = {'ok': True, 'state': 'restored'}
        return appearance.restore_result
    monkeypatch.setattr(appearance, 'restore', restore_appearance)
    monkeypatch.setattr(appearance, 'mark_rendering', lambda: None)
    environment = SimpleNamespace(
        mark_rendering=lambda: None,
        restore=lambda status, result=None: events.append(('environment', status)) or {'ok': True},
    )
    combined = run_task._combine_environment_appearance_transactions(environment, appearance)
    callbacks = []
    def submit(*, on_complete):
        callbacks.append(on_complete)
        return {'callback_ownership_transferred': True}
    result = run_task._submit_render_with_appearance(submit, combined)
    assert result['callback_ownership_transferred']
    callbacks[0]({'status': 'success', 'jobs': [], 'environment_restore_ready': True})
    assert combined.restore_result['ok']
    assert events == [('appearance', 'success'), ('environment', 'success')]
