from src.interpret.failure_taxonomy import categorize_false_negative


def test_categorizes_overexposed_as_image_quality_issue():
    category = categorize_false_negative(prob=0.4, cam_lung_overlap=0.8, mean_intensity=240)
    assert category == "image_quality_issue"


def test_categorizes_underexposed_as_image_quality_issue():
    category = categorize_false_negative(prob=0.4, cam_lung_overlap=0.8, mean_intensity=10)
    assert category == "image_quality_issue"


def test_categorizes_low_overlap_as_shortcut_driven():
    category = categorize_false_negative(prob=0.1, cam_lung_overlap=0.1, mean_intensity=128)
    assert category == "shortcut_feature_driven"


def test_categorizes_near_threshold_as_borderline():
    category = categorize_false_negative(prob=0.45, cam_lung_overlap=0.9, mean_intensity=128)
    assert category == "borderline_ground_truth"


def test_categorizes_confident_wrong_high_overlap_as_subtle_opacity():
    category = categorize_false_negative(prob=0.05, cam_lung_overlap=0.9, mean_intensity=128)
    assert category == "subtle_opacity"
