"""Tests pour le module d'anonymisation."""
import json
from pathlib import Path
import pytest
from src.anonymizer import MedicalDataAnonymizer


def test_anonymizer_init():
    """Test l'initialisation de l'anonymiseur."""
    anonymizer = MedicalDataAnonymizer()
    assert anonymizer.nlp is not None
    assert "dates" in anonymizer.nlp.pipe_names
    assert "names" in anonymizer.nlp.pipe_names
    assert "person_mentions" in anonymizer.nlp.pipe_names


def test_anonymize_text():
    """Test l'anonymisation d'un texte simple."""
    anonymizer = MedicalDataAnonymizer()
    
    # Test avec un texte contenant des informations personnelles
    text = "Le patient Jean Dupont, né le 01/01/1980, a été admis à l'hôpital Pitié-Salpêtrière."
    anonymized_text, entities = anonymizer._anonymize_text(text)
    
    # Vérifier que le texte a été modifié
    assert anonymized_text != text
    assert len(entities) > 0
    
    # Vérifier que les entités ont été détectées
    entity_types = [entity["type"] for entity in entities]
    assert any("NAME" in etype for etype in entity_types)
    assert any("DATE" in etype for etype in entity_types)


def test_anonymize_extracted_file(tmp_path):
    """Test l'anonymisation d'un fichier extrait."""
    # Créer un fichier JSON de test
    test_data = {
        "metadata": {
            "file_id": "test123",
            "processing_history": []
        },
        "pages": [{
            "page_number": 1,
            "cleaned_text": "Patient: Jean Dupont\nDate: 01/01/1980",
            "sections": {
                "diagnostic": "Le patient présente une hypertension."
            }
        }]
    }
    
    input_file = tmp_path / "test_extracted.json"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    
    # Créer le dossier de sortie
    output_dir = tmp_path / "anonymized"
    
    # Anonymiser le fichier
    anonymizer = MedicalDataAnonymizer()
    result = anonymizer.anonymize_extracted_file(input_file, output_dir)
    
    # Vérifier que le résultat contient les champs attendus
    assert "anonymized_text" in result["pages"][0]
    assert "detected_entities" in result["pages"][0]
    
    # Vérifier que le fichier de sortie a été créé
    output_file = output_dir / "test_extracted_anonymized.json"
    assert output_file.exists()


def test_process_directory(tmp_path):
    """Test le traitement d'un répertoire complet."""
    # Créer des fichiers de test
    input_dir = tmp_path / "extracted"
    output_dir = tmp_path / "anonymized"
    input_dir.mkdir()
    
    # Créer plusieurs fichiers de test
    for i in range(2):
        test_data = {
            "metadata": {"file_id": f"test{i}"},
            "pages": [{
                "page_number": 1,
                "cleaned_text": f"Patient: Jean Dupont {i}\nDate: 01/01/1980",
                "sections": {}
            }]
        }
        
        with open(input_dir / f"test{i}_extracted.json", "w", encoding="utf-8") as f:
            json.dump(test_data, f)
    
    # Traiter le répertoire
    anonymizer = MedicalDataAnonymizer()
    results = anonymizer.process_directory(input_dir, output_dir)
    
    # Vérifier les résultats
    assert len(results) == 2
    assert output_dir.exists()
    assert len(list(output_dir.glob("*_anonymized.json"))) == 2 