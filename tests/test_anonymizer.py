"""Tests pour le module d'anonymisation."""
import json
from pathlib import Path
import pytest
from src.anonymizer import MedicalDataAnonymizer


def test_anonymizer_init():
    """Test l'initialisation de l'anonymiseur."""
    anonymizer = MedicalDataAnonymizer()
    assert anonymizer.nlp is not None
    assert len(anonymizer.entity_types) > 0
    assert "ADRESSE" in anonymizer.entity_types
    assert "DATE" in anonymizer.entity_types
    assert "NOM" in anonymizer.entity_types


def test_anonymize_text():
    """Test l'anonymisation d'un texte simple."""
    anonymizer = MedicalDataAnonymizer()
    
    # Test avec un texte contenant des informations personnelles
    text = "Le patient Jean Dupont, né le 01/01/1980, a été admis à l'hôpital Pitié-Salpêtrière."
    anonymized_text, entities = anonymizer.anonymize_text(text)
    
    # Vérifier que le texte a été modifié
    assert anonymized_text != text
    assert len(entities) > 0
    
    # Vérifier que les entités ont été détectées
    entity_types = [entity["type"] for entity in entities]
    assert any(entity_type in anonymizer.entity_types for entity_type in entity_types)
    
    # Vérifier le format des remplacements
    for entity in entities:
        assert entity["replacement"].startswith("[")
        assert entity["replacement"].endswith("]")
        assert entity["type"] in entity["replacement"]
        assert "start" in entity
        assert "end" in entity
        assert "confidence" in entity


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
    
    # Vérifier que le fichier CSV des entités a été créé
    csv_file = output_dir / "test_extracted_entities.csv"
    assert csv_file.exists()
    
    # Vérifier les métadonnées
    assert "processing_history" in result["metadata"]
    assert len(result["metadata"]["processing_history"]) > 0
    last_step = result["metadata"]["processing_history"][-1]
    assert last_step["step"] == "anonymization"
    assert last_step["tool"] == "eds-pseudo"
    assert "entity_count" in last_step


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
    assert len(list(output_dir.glob("*_entities.csv"))) == 2


def test_anonymize_real_file():
    """Test l'anonymisation d'un fichier réel extrait."""
    # Chemins des répertoires
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data" / "extracted"
    output_dir = base_dir / "data" / "anonymized"
    
    # Vérifier que le répertoire d'entrée existe et contient des fichiers
    assert input_dir.exists(), f"Le répertoire {input_dir} n'existe pas"
    extracted_files = list(input_dir.glob("*_extracted.json"))
    assert len(extracted_files) > 0, f"Aucun fichier trouvé dans {input_dir}"
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Anonymiser le premier fichier trouvé
    input_file = extracted_files[0]
    anonymizer = MedicalDataAnonymizer()
    result = anonymizer.anonymize_extracted_file(input_file, output_dir)
    
    # Vérifier que le fichier anonymisé a été créé
    output_file = output_dir / f"{input_file.stem}_anonymized.json"
    assert output_file.exists(), f"Le fichier {output_file} n'a pas été créé"
    
    # Vérifier que le fichier CSV des entités a été créé
    csv_file = output_dir / f"{input_file.stem}_entities.csv"
    assert csv_file.exists(), f"Le fichier {csv_file} n'a pas été créé"
    
    # Vérifier le contenu du fichier anonymisé
    with open(output_file, 'r', encoding='utf-8') as f:
        anonymized_data = json.load(f)
    
    # Vérifier que le texte a été anonymisé
    for page in anonymized_data['pages']:
        assert 'anonymized_text' in page, "Le texte n'a pas été anonymisé"
        assert 'detected_entities' in page, "Les entités n'ont pas été détectées"
        assert len(page['detected_entities']) > 0, "Aucune entité n'a été détectée"
    
    # Vérifier les métadonnées
    assert 'metadata' in anonymized_data
    assert 'processing_history' in anonymized_data['metadata']
    assert len(anonymized_data['metadata']['processing_history']) > 0
    
    # Afficher un résumé des entités détectées
    total_entities = sum(len(page['detected_entities']) for page in anonymized_data['pages'])
    print(f"\nRésumé de l'anonymisation :")
    print(f"- Fichier traité : {input_file.name}")
    print(f"- Nombre total d'entités détectées : {total_entities}")
    print(f"- Types d'entités trouvés : {set(entity['type'] for page in anonymized_data['pages'] for entity in page['detected_entities'])}") 