"""Tests pour le module d'extraction PDF."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from src.pdf_extractor import PDFExtractor


class MockPDFExtractor(PDFExtractor):
    """Version modifiée de PDFExtractor pour les tests."""
    def __init__(self, *args, **kwargs):
        """Surcharge l'initialisation pour éviter la vérification du répertoire."""
        self.input_dir = Path("dummy_path")
        self.logger = MagicMock()
        self.ocr_lang = kwargs.get('ocr_lang', 'fra')


def test_pdf_extractor_init(tmp_path):
    """Test l'initialisation de PDFExtractor."""
    # Test avec un répertoire existant
    extractor = PDFExtractor(tmp_path)
    assert extractor.input_dir == tmp_path

    # Test avec un répertoire inexistant
    with pytest.raises(FileNotFoundError):
        PDFExtractor("dossier_inexistant")

    # Test avec fichier de log
    log_file = tmp_path / "extraction.log"
    extractor_with_log = PDFExtractor(tmp_path, log_file=str(log_file))
    assert log_file.exists()

    # Test avec configuration OCR
    extractor_with_ocr = PDFExtractor(tmp_path, tesseract_path="/usr/local/bin/tesseract", ocr_lang="fra+eng")
    assert extractor_with_ocr.ocr_lang == "fra+eng"


def test_clean_text():
    """Test le nettoyage du texte."""
    extractor = MockPDFExtractor()
    
    # Test de normalisation des espaces
    text = "Texte   avec    des   espaces  multiples."
    cleaned = extractor._clean_text(text)
    assert cleaned == "Texte avec des espaces multiples."

    # Test de séparation des phrases
    text = "Première phrase. Deuxième phrase. Troisième phrase."
    cleaned = extractor._clean_text(text)
    assert len(cleaned.split('\n')) == 3


def test_identify_sections():
    """Test l'identification des sections médicales."""
    extractor = MockPDFExtractor()
    
    text = """
    ANTÉCÉDENTS MÉDICAUX
    Patient diabétique type 2.
    
    DIAGNOSTIC
    Hypertension artérielle.
    
    TRAITEMENT
    Metformine 1000mg.
    """
    
    sections = extractor._identify_sections(text)
    assert 'antecedents' in sections
    assert 'diagnostic' in sections
    assert 'traitement' in sections
    assert 'diabétique type 2' in sections['antecedents'].lower()


def test_is_scanned_pdf():
    """Test la détection de pages scannées."""
    extractor = MockPDFExtractor()
    
    # Test avec une page vide (scannée)
    assert extractor._is_scanned_pdf("") is True
    
    # Test avec peu de texte (probablement scannée)
    assert extractor._is_scanned_pdf("ABC") is True
    
    # Test avec du texte normal (non scannée)
    long_text = "Ceci est un long texte médical qui contient suffisamment de caractères pour être considéré comme du texte extrait normalement."
    assert extractor._is_scanned_pdf(long_text) is False


@patch('src.pdf_extractor.convert_from_path')
@patch('src.pdf_extractor.pytesseract.image_to_data')
def test_perform_ocr(mock_image_to_data, mock_convert_from_path, tmp_path):
    """Test l'OCR sur une page."""
    extractor = MockPDFExtractor()
    
    # Simuler une image PDF
    mock_image = MagicMock()
    mock_convert_from_path.return_value = [mock_image]
    
    # Simuler les résultats OCR
    mock_image_to_data.return_value = {
        'text': ['Ceci', 'est', 'un', 'test', 'OCR'],
        'conf': [90.5, 85.2, 95.0, 88.7, 92.1]
    }
    
    # Créer un fichier PDF factice
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()
    
    # Tester l'OCR
    text, confidence = extractor._perform_ocr(pdf_path, 1)
    assert text == "Ceci est un test OCR"
    assert confidence == pytest.approx(90.3, 0.1)  # Moyenne des scores de confiance


def test_extract_text_from_file(tmp_path):
    """Test l'extraction de texte d'un fichier PDF."""
    # Utilisation du fichier PDF réel
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data/raw"
    extracted_dir = base_dir / "data/extracted"
    extracted_dir.mkdir(exist_ok=True)
    
    extractor = PDFExtractor(input_dir)
    
    # Test avec le fichier PDF réel
    pdf_path = input_dir / "test.pdf"
    assert pdf_path.exists(), f"Le fichier {pdf_path} n'existe pas"
    
    result = extractor.extract_text_from_file(pdf_path)
    assert result is not None
    assert isinstance(result, dict)
    assert 'metadata' in result
    assert 'pages' in result
    assert len(result['pages']) > 0
    
    # Sauvegarde du résultat
    output_file = extracted_dir / f"{pdf_path.stem}_extracted.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Vérification du contenu des pages
    first_page = result['pages'][0]
    assert isinstance(first_page, dict)
    assert 'cleaned_text' in first_page
    assert isinstance(first_page['cleaned_text'], str)
    assert len(first_page['cleaned_text']) > 0
    
    # Vérification des attributs spécifiques
    assert 'has_tables' in first_page
    assert isinstance(first_page['has_tables'], bool)
    assert 'is_scanned' in first_page
    assert isinstance(first_page['is_scanned'], bool)
    assert 'ocr_confidence' in first_page
    
    # Test avec un fichier inexistant
    with pytest.raises(FileNotFoundError):
        extractor.extract_text_from_file("fichier_inexistant.pdf")
    
    # Test avec un fichier non-PDF
    non_pdf = tmp_path / "test.txt"
    non_pdf.touch()
    with pytest.raises(ValueError):
        extractor.extract_text_from_file(non_pdf)


def test_process_directory(tmp_path):
    """Test le traitement d'un répertoire."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    
    extractor = PDFExtractor(input_dir)
    results = extractor.process_directory(output_dir)
    
    assert isinstance(results, dict)
    assert output_dir.exists()


def test_metadata_extraction(tmp_path):
    """Test l'extraction des métadonnées."""
    extractor = MockPDFExtractor()
    
    # Vérification de la structure des métadonnées
    class MockPdfReader:
        def __init__(self):
            self.metadata = {
                '/CreationDate': '20240101',
                '/Author': 'Dr Test'
            }
            self.pages = [None]  # Simuler une page

    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()
    
    metadata = extractor._extract_metadata(MockPdfReader(), pdf_path, is_scanned=True)
    
    assert metadata.filename == "test.pdf"
    assert metadata.author == "Dr Test"
    assert metadata.creation_date == "20240101"
    assert metadata.page_count == 1
    assert isinstance(metadata.extraction_date, str)
    assert isinstance(metadata.processing_history, list)
    assert len(metadata.processing_history) == 1
    assert metadata.is_scanned is True
    assert metadata.ocr_applied is True 