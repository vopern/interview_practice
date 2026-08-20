from src.models.interview import InterviewSettings
from src.storage.interview_storage import InterviewStorage
from tests.test_models import make_session


def test_save_and_load_roundtrip(tmp_path):
    storage = InterviewStorage(str(tmp_path))
    session = make_session()
    path = storage.save_session(session)
    assert path.exists()

    loaded = storage.load_sessions(session.settings)
    assert len(loaded) == 1
    assert loaded[0].to_dict() == session.to_dict()


def test_load_sessions_empty_for_unknown_settings(tmp_path):
    storage = InterviewStorage(str(tmp_path))
    assert (
        storage.load_sessions(InterviewSettings("Nobody", "Nothing", "HR Screen")) == []
    )


def test_sessions_sorted_oldest_first(tmp_path):
    storage = InterviewStorage(str(tmp_path))
    first = make_session()
    first.id = "20260101_090000"
    second = make_session()
    second.id = "20260201_090000"
    storage.save_session(second)
    storage.save_session(first)
    loaded = storage.load_sessions(first.settings)
    assert [s.id for s in loaded] == ["20260101_090000", "20260201_090000"]


def test_load_sessions_before_cuts_off_at_the_named_session(tmp_path):
    """What a replayed round is shown: the history its source ran with, not itself."""
    storage = InterviewStorage(str(tmp_path))
    for session_id in ("20260101_090000", "20260201_090000", "20260301_090000"):
        session = make_session()
        session.id = session_id
        storage.save_session(session)

    settings = make_session().settings
    loaded = storage.load_sessions(settings, before="20260201_090000")
    assert [s.id for s in loaded] == ["20260101_090000"]
    assert len(storage.load_sessions(settings)) == 3
    assert storage.load_sessions(settings, before="20260101_090000") == []


def test_list_slugs(tmp_path):
    storage = InterviewStorage(str(tmp_path))
    assert storage.list_slugs() == []
    session = make_session()
    storage.save_session(session)
    slugs = storage.list_slugs()
    assert len(slugs) == 1
    slug, settings = slugs[0]
    assert slug == session.settings.slug
    assert settings.company == "Acme Corp"


def test_load_company_context_concatenates_files(tmp_path):
    storage = InterviewStorage(str(tmp_path))
    folder = tmp_path / "context" / "acme_corp"
    folder.mkdir(parents=True)
    (folder / "research.md").write_text("They build rockets.", encoding="utf-8")
    (folder / "recruiter_notes.txt").write_text("Panel of two.", encoding="utf-8")
    (folder / ".gitkeep").write_text("", encoding="utf-8")

    context = storage.load_company_context("Acme Corp")  # case/punctuation-insensitive
    assert "--- research.md ---" in context
    assert "They build rockets." in context
    assert "Panel of two." in context
    assert ".gitkeep" not in context


def test_load_company_context_empty_when_no_match(tmp_path):
    storage = InterviewStorage(str(tmp_path))
    assert storage.load_company_context("Acme Corp") == ""
    (tmp_path / "context" / "other_co").mkdir(parents=True)
    assert storage.load_company_context("Acme Corp") == ""


def test_corrupt_file_is_skipped(tmp_path):
    storage = InterviewStorage(str(tmp_path))
    session = make_session()
    storage.save_session(session)
    bad = storage._slug_dir(session.settings.slug) / "session_broken.json"
    bad.write_text("{not json", encoding="utf-8")
    loaded = storage.load_sessions(session.settings)
    assert len(loaded) == 1


def test_save_company_context_creates_a_normalized_folder(tmp_path):
    storage = InterviewStorage(str(tmp_path))
    path = storage.save_company_context("ABC Corp", "research_1.md", "They trade.")

    assert path == tmp_path / "context" / "abc_corp" / "research_1.md"
    assert "They trade." in storage.load_company_context("ABC Corp")


def test_save_company_context_reuses_an_existing_folder(tmp_path):
    """A new report must land where the planner already reads, not beside it."""
    storage = InterviewStorage(str(tmp_path))
    folder = tmp_path / "context" / "acme_corp"
    folder.mkdir(parents=True)
    (folder / "recruiter_notes.txt").write_text("Panel of two.", encoding="utf-8")

    path = storage.save_company_context("Acme  Corp!", "research_1.md", "Rockets.")

    assert path.parent == folder
    assert [p.name for p in storage.list_context_files("acme corp")] == [
        "recruiter_notes.txt",
        "research_1.md",
    ]
    context = storage.load_company_context("Acme Corp")
    assert "Panel of two." in context and "Rockets." in context


def test_list_context_companies_skips_folders_with_only_hidden_files(tmp_path):
    storage = InterviewStorage(str(tmp_path))
    (tmp_path / "context" / "empty_co").mkdir(parents=True)
    (tmp_path / "context" / "empty_co" / ".gitkeep").write_text("", encoding="utf-8")
    storage.save_company_context("Acme", "research_1.md", "Rockets.")

    assert storage.list_context_companies() == ["Acme"]


def test_listed_company_names_still_find_their_own_folder(tmp_path):
    """The name the picker offers has to resolve back to the folder it came from.

    It is the readable form, not the folder name, because it goes straight into
    `InterviewSettings.company` and from there into every prompt.
    """
    storage = InterviewStorage(str(tmp_path))
    storage.save_company_context("example_labs", "research_1.md", "They deliver.")

    assert storage.list_context_companies() == ["Example Labs"]
    listed = storage.list_context_companies()[0]
    assert [p.name for p in storage.list_context_files(listed)] == ["research_1.md"]
    assert "They deliver." in storage.load_company_context(listed)


def test_read_context_file_returns_empty_for_undecodable_files(tmp_path):
    storage = InterviewStorage(str(tmp_path))
    path = storage.save_company_context("Acme", "logo.bin", "")
    path.write_bytes(b"\xff\xfe\x00binary")

    assert storage.read_context_file(path) == ""
    # One bad drop-in must not take the whole folder down with it.
    storage.save_company_context("Acme", "research_1.md", "Rockets.")
    assert "Rockets." in storage.load_company_context("Acme")
