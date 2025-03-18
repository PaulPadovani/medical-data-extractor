"""
Module pour l'extraction de texte à partir de fichiers PDF médicaux.
"""
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, List, Any, Tuple

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader
from tqdm import tqdm


@dataclass
class PDFMetadata:
    """Métadonnées d'un document PDF."""
    file_id: str
    filename: str
    creation_date: Optional[str]
    modification_date: Optional[str]
    author: Optional[str]
    producer: Optional[str]
    page_count: int
    extraction_date: str
    processing_history: List[Dict[str, Any]]
    is_scanned: bool
    ocr_applied: bool


@dataclass
class ProcessedPage:
    """Page traitée avec son texte et ses métadonnées."""
    page_number: int
    raw_text: str
    cleaned_text: str
    sections: Dict[str, str]
    has_tables: bool
    processing_notes: List[str]
    is_scanned: bool
    ocr_confidence: Optional[float]


class PDFExtractor:
    """Classe pour extraire le texte des fichiers PDF médicaux."""

    # Patterns pour le nettoyage du texte
    CLEANUP_PATTERNS = [
        (r'\s+', ' '),  # Normalisation des espaces
        (r'(?<=[.!?])\s+(?=[A-Z])', '\n'),  # Séparation des phrases
    ]

    # Patterns pour identifier les sections médicales communes
    SECTION_PATTERNS = {
        'antecedents': r'(?i)ant[ée]c[ée]dents?|histoire\s+m[ée]dicale',
        'diagnostic': r'(?i)diagnostic|conclusion|impression\s+clinique',
        'traitement': r'(?i)traitement|prescription|ordonnance',
        'examen_clinique': r'(?i)examen\s+clinique|observation|status',
    }

    def __init__(self, input_dir: Union[str, Path], log_file: Optional[str] = None,
                 tesseract_path: Optional[str] = None, ocr_lang: str = 'fra'):
        """
        Initialise l'extracteur PDF.

        Args:
            input_dir: Chemin vers le répertoire contenant les PDF à traiter
            log_file: Chemin vers le fichier de log (optionnel)
            tesseract_path: Chemin vers l'exécutable Tesseract (optionnel)
            ocr_lang: Langue pour l'OCR (défaut: français)
        """
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Le répertoire {input_dir} n'existe pas")

        # Configuration du logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)

        # Configuration OCR
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.ocr_lang = ocr_lang

    def _generate_file_id(self, file_path: Path, content: bytes) -> str:
        """Génère un identifiant unique pour le fichier."""
        hasher = hashlib.sha256()
        hasher.update(str(file_path).encode())
        hasher.update(content)
        return hasher.hexdigest()[:16]

    def _clean_text(self, text: str) -> str:
        """Nettoie et normalise le texte extrait."""
        cleaned = text
        for pattern, replacement in self.CLEANUP_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned.strip()

    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Identifie les sections médicales dans le texte."""
        sections = {}
        current_text = text
        
        for section_name, pattern in self.SECTION_PATTERNS.items():
            matches = list(re.finditer(pattern, current_text, re.IGNORECASE | re.MULTILINE))
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(current_text)
                section_content = current_text[start:end].strip()
                sections[section_name] = section_content
                
        return sections

    def _is_scanned_pdf(self, page_text: str) -> bool:
        """
        Détermine si une page est probablement scannée en vérifiant la quantité de texte extraite.
        
        Args:
            page_text: Texte extrait de la page
            
        Returns:
            bool: True si la page semble être scannée
        """
        # Si le texte est vide ou contient très peu de caractères, c'est probablement une page scannée
        return len(page_text.strip()) < 100

    def _perform_ocr(self, pdf_path: Path, page_number: int) -> Tuple[str, float]:
        """
        Effectue l'OCR sur une page spécifique d'un PDF.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            page_number: Numéro de la page (1-indexed)
            
        Returns:
            Tuple[str, float]: Texte extrait et score de confiance
        """
        try:
            # Convertir la page PDF en image
            images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
            if not images:
                return "", 0.0

            image = images[0]
            
            # Effectuer l'OCR
            ocr_data = pytesseract.image_to_data(image, lang=self.ocr_lang, output_type=pytesseract.Output.DICT)
            
            # Extraire le texte et calculer la confiance moyenne
            text_parts = []
            confidence_scores = []
            
            for i, conf in enumerate(ocr_data['conf']):
                if conf > 0:  # Ignorer les valeurs de confiance négatives
                    text = ocr_data['text'][i]
                    if text.strip():
                        text_parts.append(text)
                        confidence_scores.append(conf)
            
            extracted_text = ' '.join(text_parts)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            return extracted_text, avg_confidence

        except Exception as e:
            self.logger.error(f"Erreur lors de l'OCR de la page {page_number}: {str(e)}")
            return "", 0.0

    def _extract_metadata(self, pdf_reader: PdfReader, file_path: Path, is_scanned: bool = False) -> PDFMetadata:
        """Extrait les métadonnées du PDF."""
        info = pdf_reader.metadata
        with open(file_path, 'rb') as f:
            content = f.read()
            
        return PDFMetadata(
            file_id=self._generate_file_id(file_path, content),
            filename=file_path.name,
            creation_date=info.get('/CreationDate', None),
            modification_date=info.get('/ModDate', None),
            author=info.get('/Author', None),
            producer=info.get('/Producer', None),
            page_count=len(pdf_reader.pages),
            extraction_date=datetime.now().isoformat(),
            processing_history=[{
                'step': 'extraction',
                'timestamp': datetime.now().isoformat(),
                'details': 'Extraction initiale du texte'
            }],
            is_scanned=is_scanned,
            ocr_applied=is_scanned
        )

    def extract_text_from_file(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extrait le texte et les métadonnées d'un fichier PDF.

        Args:
            pdf_path: Chemin vers le fichier PDF

        Returns:
            Dict[str, Any]: Dictionnaire contenant les métadonnées et le texte extrait

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le fichier n'est pas un PDF valide
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Le fichier {pdf_path} n'existe pas")
        
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"Le fichier {pdf_path} n'est pas un PDF")

        try:
            reader = PdfReader(pdf_path)
            processed_pages = []
            is_scanned = False

            # Premier passage : extraction standard et détection des pages scannées
            for page_num, page in enumerate(reader.pages, 1):
                raw_text = page.extract_text()
                page_is_scanned = self._is_scanned_pdf(raw_text)
                is_scanned = is_scanned or page_is_scanned

                if page_is_scanned:
                    # Si la page est scannée, effectuer l'OCR
                    raw_text, confidence = self._perform_ocr(pdf_path, page_num)
                    processing_note = f"OCR appliqué avec une confiance de {confidence:.2f}%"
                else:
                    confidence = None
                    processing_note = "Extraction standard du texte"

                if raw_text.strip():
                    cleaned_text = self._clean_text(raw_text)
                    sections = self._identify_sections(cleaned_text)
                    has_tables = bool(re.search(r'\|\s*\w+\s*\|', raw_text))

                    processed_page = ProcessedPage(
                        page_number=page_num,
                        raw_text=raw_text,
                        cleaned_text=cleaned_text,
                        sections=sections,
                        has_tables=has_tables,
                        processing_notes=[processing_note],
                        is_scanned=page_is_scanned,
                        ocr_confidence=confidence
                    )
                    processed_pages.append(asdict(processed_page))

            metadata = self._extract_metadata(reader, pdf_path, is_scanned)
            return {
                'metadata': asdict(metadata),
                'pages': processed_pages
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction du PDF {pdf_path}: {str(e)}")
            raise ValueError(f"Erreur lors de l'extraction du PDF {pdf_path}: {str(e)}")

    def process_directory(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Traite tous les fichiers PDF dans le répertoire d'entrée.

        Args:
            output_dir: Répertoire optionnel pour sauvegarder les textes extraits

        Returns:
            Dict[str, Dict[str, Any]]: Dictionnaire avec les identifiants de fichiers comme clés
                                      et les données extraites comme valeurs
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        for pdf_file in tqdm(pdf_files, desc="Extraction des PDF"):
            try:
                extracted_data = self.extract_text_from_file(pdf_file)
                file_id = extracted_data['metadata']['file_id']
                results[file_id] = extracted_data

                if output_dir:
                    # Sauvegarde du texte extrait
                    text_file = output_dir / f"{file_id}.txt"
                    with open(text_file, 'w', encoding='utf-8') as f:
                        for page in extracted_data['pages']:
                            f.write(f"=== Page {page['page_number']} ===\n")
                            f.write(f"Texte nettoyé:\n{page['cleaned_text']}\n\n")
                            if page['sections']:
                                f.write("Sections détectées:\n")
                                for section, content in page['sections'].items():
                                    f.write(f"--- {section} ---\n{content}\n\n")
                            if page['is_scanned']:
                                f.write(f"Note: Page scannée - OCR appliqué (confiance: {page['ocr_confidence']:.2f}%)\n\n")

                    # Sauvegarde des métadonnées
                    meta_file = output_dir / f"{file_id}_meta.json"
                    with open(meta_file, 'w', encoding='utf-8') as f:
                        json.dump(extracted_data['metadata'], f, indent=2, ensure_ascii=False)

            except Exception as e:
                self.logger.error(f"Erreur lors du traitement de {pdf_file}: {str(e)}")
                continue

        return results
