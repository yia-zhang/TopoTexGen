"""The queue's job: several workers divide the work without talking, and no
object is ever done twice or lost."""
import json

import pytest

from topotexgen.workqueue import WorkQueue

UIDS = [f"{i:032x}" for i in range(8)]
KEY = "recipe-1"


@pytest.fixture
def root(tmp_path):
    return tmp_path / "queue"


def test_two_workers_never_claim_the_same_object(root):
    a, b = WorkQueue(root, "a"), WorkQueue(root, "b")
    ca = [c.uid for c in a.iter_work(UIDS, lambda u: KEY)]
    cb = [c.uid for c in b.iter_work(UIDS, lambda u: KEY)]
    assert set(ca) & set(cb) == set()
    assert set(ca) | set(cb) == set(UIDS)


def test_a_completed_object_is_skipped_on_the_next_pass(root):
    q = WorkQueue(root, "a")
    first = next(q.iter_work(UIDS, lambda u: KEY))
    q.complete(first.uid, KEY)
    again = [c.uid for c in WorkQueue(root, "a").iter_work(UIDS, lambda u: KEY)]
    assert first.uid not in again


def test_a_recipe_change_makes_finished_work_pending_again(root):
    """This is what stops a stale product being served: done means done UNDER
    THIS KEY, not merely present."""
    q = WorkQueue(root, "a")
    for c in list(q.iter_work(UIDS, lambda u: KEY)):
        q.complete(c.uid, KEY)
    assert q.status(UIDS, lambda u: KEY)["done"] == len(UIDS)
    s = q.status(UIDS, lambda u: "recipe-2")
    assert s["done"] == 0 and s["superseded_by_recipe"] == len(UIDS)


def test_a_released_object_returns_to_the_pool(root):
    q = WorkQueue(root, "a")
    c = next(q.iter_work(UIDS, lambda u: KEY))
    q.release(c.uid)
    assert c.uid in [x.uid for x in WorkQueue(root, "b").iter_work(UIDS, lambda u: KEY)]


def test_a_dead_workers_claim_expires_so_the_object_is_not_lost(root):
    """A killed worker must not park an object forever."""
    dead = WorkQueue(root, "dead", lease_s=0)
    c = next(dead.iter_work(UIDS, lambda u: KEY))
    live = WorkQueue(root, "live", lease_s=0)
    assert c.uid in [x.uid for x in live.iter_work(UIDS, lambda u: KEY)]


def test_a_fresh_claim_is_respected_by_other_workers(root):
    fresh = WorkQueue(root, "holder", lease_s=3600)
    c = next(fresh.iter_work(UIDS, lambda u: KEY))
    other = WorkQueue(root, "other", lease_s=3600)
    assert c.uid not in [x.uid for x in other.iter_work(UIDS, lambda u: KEY)]


def test_completion_records_who_did_it_and_under_which_key(root):
    q = WorkQueue(root, "w7")
    q.complete(UIDS[0], KEY, seconds=12.5)
    rec = json.loads((root / "done" / f"{UIDS[0]}.json").read_text())
    assert rec["owner"] == "w7" and rec["key"] == KEY and rec["seconds"] == 12.5


def test_limit_bounds_how_much_one_worker_takes(root):
    q = WorkQueue(root, "a")
    assert len(list(q.iter_work(UIDS, lambda u: KEY, limit=3))) == 3


def test_a_stage_can_read_back_what_the_previous_one_derived_its_product_from(tmp_path):
    """A key says which INPUTS a product claims. It cannot say whether those
    inputs are still the bytes on disk, so a stage records the digests it
    derived from and the next stage reads them back. Existence is not
    provenance, and a key alone is not either: a generator can complete twice
    at the same key -- a crash and a retry -- leaving different bytes behind
    the same marker.
    """
    q = WorkQueue(tmp_path / "q", "writer")
    q.complete("u1", "key-A", atlas_digest="aaaa1111", delivery_key="d-1")
    assert q.completed_key("u1") == "key-A"
    extra = q.completed_extra("u1")
    assert extra == {"atlas_digest": "aaaa1111", "delivery_key": "d-1"}
    # the bookkeeping fields are not part of what a consumer reads back
    assert "owner" not in extra and "at" not in extra
    assert q.completed_extra("never-seen") is None
