from coffea.nanoevents.schemas.schema import auto_schema


def test_auto_empty():
    "Test an empy incoming form"

    b = auto_schema({"contents": {}})

    assert len(b.form["contents"]) == 0


def test_auto_single_no_structure():
    "Test a single item with no underscore"

    b = auto_schema({"contents": {"pt": {}}})

    assert len(b.form["contents"]) == 1


def test_auto_single_with_structure():
    "Test a single item, that does have an underscore"

    b = auto_schema(
        {
            "contents": {
                "lep_pt": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
            },
        }
    )

    assert "lep" in b.form["contents"]
    assert "pt" in b.form["contents"]["lep"]["content"]["contents"]


def test_auto_multi_with_structure():
    "Test a multiple items, that does have an underscore"

    b = auto_schema(
        {
            "contents": {
                "lep_pt": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "lep_eta": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "lep_phi": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
            },
        }
    )

    assert "lep" in b.form["contents"]
    assert "pt" in b.form["contents"]["lep"]["content"]["contents"]
    assert "phi" in b.form["contents"]["lep"]["content"]["contents"]
    assert "eta" in b.form["contents"]["lep"]["content"]["contents"]


def test_auto_multi_and_single_with_structure():
    "Test a multiple items along with a non-structure item, that does have an underscore"

    b = auto_schema(
        {
            "contents": {
                "lep_pt": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "lep_eta": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "lep_phi": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "fork": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
            },
        }
    )

    assert "lep" in b.form["contents"]
    assert "fork" in b.form["contents"]
    assert (
        b.form["contents"]["lep"]["content"]["parameters"]["__record__"]
        == "NanoCollection"
    )


def test_auto_4vector_mass():
    "Test a multiple items, that does have an underscore"

    b = auto_schema(
        {
            "contents": {
                "lep_pt": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "lep_eta": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "lep_phi": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "lep_mass": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "lep_charge": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                # Make sure we can handle an extra item besides!
                "lep_extra": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
            },
        }
    )

    assert "lep" in b.form["contents"]
    assert "extra" in b.form["contents"]["lep"]["content"]["contents"]
    assert (
        b.form["contents"]["lep"]["content"]["parameters"]["__record__"]
        == "PtEtaPhiMCandidate"
    )


def test_auto_4vector_e():
    "Test a multiple items, that does have an underscore"

    b = auto_schema(
        {
            "contents": {
                "lep_pt": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "lep_eta": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "lep_phi": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "lep_energy": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                "lep_charge": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
                # Make sure we can handle an extra item besides!
                "lep_extra": {
                    "content": {"class": "RecordArray"},
                    "class": "RecordArray",
                    "offsets": None,
                    "form_key": None,
                },
            },
        }
    )

    assert "lep" in b.form["contents"]
    assert "extra" in b.form["contents"]["lep"]["content"]["contents"]
    assert (
        b.form["contents"]["lep"]["content"]["parameters"]["__record__"]
        == "PtEtaPhiECandidate"
    )
