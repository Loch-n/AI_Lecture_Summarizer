import pytest
import tempfile
import os
from pathlib import Path
from app import (
    format_timestamp,
    extract_key_phrases,
    setup_spacy,
    check_ffmpeg
)

def test_format_timestamp():
    """Test timestamp formatting function"""
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(60) == "01:00"
    assert format_timestamp(125) == "02:05"
    assert format_timestamp(3661) == "61:01"

def test_extract_key_phrases():
    """Test key phrase extraction"""
    text = "Machine learning is a powerful tool for data analysis and artificial intelligence applications."
    phrases = extract_key_phrases(text, top_n=3)
    assert isinstance(phrases, list)
    assert len(phrases) &lt;= 3

def test_setup_spacy():
    """Test spaCy setup"""
    nlp = setup_spacy()
    # Should either return a spaCy model or None
    assert nlp is None or hasattr(nlp, '__call__')

def test_check_ffmpeg():
    """Test FFmpeg availability check"""
    result = check_ffmpeg()
    assert isinstance(result, bool)

def test_app_imports():
    """Test that the app can be imported without errors"""
    try:
        import app
        assert hasattr(app, 'run_pipeline')
        assert hasattr(app, 'create_interface')
    except ImportError as e:
        pytest.fail(f"Failed to import app: {e}")

if __name__ == "__main__":
    pytest.main([__file__])