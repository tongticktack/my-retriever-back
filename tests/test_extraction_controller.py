import pytest
from datetime import datetime, timedelta

from app.services import extraction_controller as extc
from app.services import lost_item_extractor as li_ext


def _make_state():
    return {"items": [{"extracted": {}, "missing": li_ext.compute_missing({}), "stage": "collecting"}], "active_index": 0}


def test_rule_then_llm_trigger_by_missing(monkeypatch):
    state = _make_state()
    # monkeypatch LLM strict extract to return deterministic JSON
    called = {}
    def fake_llm(user_text, current):
        called['ran'] = True
        return {"category": "가방", "subcategory": "백팩", "color": "검정", "region": "강남", "lost_date": (datetime.now().date() - timedelta(days=1)).isoformat()}
    monkeypatch.setattr(extc, '_strict_llm_extract', fake_llm)
    reply, new_state, model, snap = extc.process_message("어제 강남에서 검정 백팩 잃어버렸어요", state, False)
    assert reply is not None
    assert '강남' in str(snap['extracted'])
    assert called.get('ran') is True


def test_date_ambiguity(monkeypatch):
    state = _make_state()
    def fake_llm(user_text, current):
        return {}
    monkeypatch.setattr(extc, '_strict_llm_extract', fake_llm)
    # Provide ambiguous relative refs (simulate by crafting two patterns)
    # Controller will run extraction; rule extraction currently may not produce multi-candidates easily, so emulate by inserting.
    state['items'][0]['extracted'] = {"lost_date_candidates": ",".join([
        (datetime.now().date() - timedelta(days=1)).isoformat(),
        (datetime.now().date() - timedelta(days=2)).isoformat(),
    ])}
    state['items'][0]['missing'] = li_ext.compute_missing({})
    reply, new_state, model, snap = extc.process_message("어제나 그제 같아요", state, False)
    assert '날짜가 모호합니다' in reply


def test_confirmation_flow(monkeypatch):
    state = _make_state()
    # Fill all fields to trigger ready state next
    filled = {"category": "가방", "subcategory": "백팩", "color": "검정", "lost_date": datetime.now().date().isoformat(), "region": "강남"}
    state['items'][0]['extracted'] = filled.copy()
    state['items'][0]['missing'] = []
    # First pass should move to ready
    reply, state, model, snap = extc.process_message("네", state, False)
    # If already ready and user says '예' then confirm, so ensure stage transitions
    state['items'][0]['stage'] = 'ready'
    reply, state, model, snap = extc.process_message("예", state, False)
    assert state['items'][0]['stage'] == 'confirmed'


def test_multi_color_collection(monkeypatch):
    state = _make_state()
    # LLM bypass: ensure rule extractor sees multiple colors
    def fake_llm(user_text, current):
        return {}
    monkeypatch.setattr(extc, '_strict_llm_extract', fake_llm)
    reply, state, model, snap = extc.process_message("파란 빨간 검은 가방을 잃어버렸어요", state, False)
    colors_all = snap['extracted'].get('color_all')
    assert colors_all is not None
    # representative should be first by priority list order present among colors
    assert snap['extracted'].get('color') in colors_all.split(',')
