import json
import logging
import csv
import concurrent.futures
from pathlib import Path
import edsnlp
import huggingface_hub
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class Pseudonymizer:
    """Classe pour pseudonymiser les textes extraits des PDF médicaux en utilisant eds-pseudo."""
    
    def __init__(self, model: str = "AP-HP/eds-pseudo-public"):
        """Initialise le pipeline NLP avec eds-pseudo."""
        # Récupérer le token depuis les variables d'environnement
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("Le token Hugging Face n'est pas configuré. Créez un fichier .env avec HUGGINGFACE_TOKEN=votre_token")
        
        huggingface_hub.login(token=token)
        self.nlp = edsnlp.load(model, auto_update=True)
        
    def pseudonymize_text(self, text: str) -> tuple[str, list]:
        """Applique la pseudonymisation sur un texte médical et retourne le texte modifié avec un tableau des entités remplacées."""
        doc = self.nlp(text)
        anonymized_text = text
        pseudonymization_table = []
        
        for ent in doc.ents:
            replacement = f"[ANONYMIZED_{ent.label_}]"
            pseudonymization_table.append({"original": ent.text, "replacement": replacement, "label": ent.label_})
            anonymized_text = anonymized_text.replace(ent.text, replacement)
        
        return anonymized_text, pseudonymization_table
    
    def process_single_file(self, file: Path, input_dir: Path, output_dir: Path):
        """Traite un seul fichier pour la pseudonymisation et enregistre les résultats."""
        with open(file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        text_file = input_dir / f"{metadata['file_id']}.txt"
        if not text_file.exists():
            logging.warning(f"Fichier texte manquant pour {file}")
            return
        
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        pseudonymized_text, pseudonymization_table = self.pseudonymize_text(text)
        
        output_file = output_dir / f"{metadata['file_id']}_pseudonymized.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(pseudonymized_text)
        
        table_file_json = output_dir / f"{metadata['file_id']}_pseudonymization_table.json"
        with open(table_file_json, "w", encoding="utf-8") as f:
            json.dump(pseudonymization_table, f, indent=2, ensure_ascii=False)
        
        table_file_csv = output_dir / f"{metadata['file_id']}_pseudonymization_table.csv"
        with open(table_file_csv, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Original", "Replacement", "Label"])
            for row in pseudonymization_table:
                writer.writerow([row["original"], row["replacement"], row["label"]])
        
    def process_extracted_files(self, input_dir: Path, output_dir: Path):
        """Traite tous les fichiers extraits en parallèle pour appliquer la pseudonymisation."""
        input_dir = Path(input_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = list(input_dir.glob("*_meta.json"))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_single_file, file, input_dir, output_dir) for file in files]
            concurrent.futures.wait(futures)
        
        logging.info("Pseudonymisation terminée pour tous les fichiers.")
    
    def get_pseudonymized_text(self, text: str) -> str:
        """Retourne le texte pseudonymisé sans traitement de fichier."""
        pseudonymized_text, _ = self.pseudonymize_text(text)
        return pseudonymized_text

class MedicalDataAnonymizer:
    """Classe pour anonymiser les données médicales extraites."""

    def __init__(self):
        """Initialise le pipeline d'anonymisation avec eds-nlp."""
        # Créer le pipeline eds-nlp avec les composants nécessaires
        self.nlp = edsnlp.blank("eds")
        
        # Ajouter les composants pour la détection des entités
        self.nlp.add_pipe("normalizer")
        self.nlp.add_pipe("sentences")
        
        # Composants pour l'anonymisation des données personnelles
        self.nlp.add_pipe("names", config={"mode": "strict"})
        self.nlp.add_pipe("dates", config={"mode": "strict"})
        self.nlp.add_pipe("person_mentions", config={"mode": "strict"})
        
        # Composants pour l'anonymisation des données médicales
        self.nlp.add_pipe("hospital_names", config={"mode": "strict"})
        self.nlp.add_pipe("medican_mentions")  # Pour les mentions médicales
        self.nlp.add_pipe("measurements")  # Pour les mesures et valeurs
        
        # Composants pour les données sensibles supplémentaires
        self.nlp.add_pipe("addresses")  # Pour les adresses
        self.nlp.add_pipe("ids")  # Pour les identifiants

    def _anonymize_text(self, text: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Anonymise un texte donné en utilisant eds-nlp.

        Args:
            text: Texte à anonymiser

        Returns:
            tuple: (texte anonymisé, liste des entités détectées)
        """
        doc = self.nlp(text)
        anonymized_text = text
        entities = []

        # Collecter toutes les entités détectées (dans l'ordre inverse pour préserver les indices)
        for ent in reversed(doc.ents):
            entity_type = ent.label_
            entity_text = ent.text
            start, end = ent.start_char, ent.end_char
            
            # Créer un identifiant de remplacement plus descriptif
            replacement = f"[{entity_type}_{len(entities):03d}]"
            
            # Remplacer l'entité dans le texte
            anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
            
            # Sauvegarder l'information sur l'entité avec plus de détails
            entities.append({
                "type": entity_type,
                "text": entity_text,
                "start": start,
                "end": end,
                "replacement": replacement,
                "confidence": getattr(ent, "confidence", None),
                "pattern": getattr(ent, "pattern", None)
            })

        return anonymized_text, entities

    def anonymize_extracted_file(self, input_file: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Anonymise un fichier JSON extrait.

        Args:
            input_file: Chemin vers le fichier JSON extrait
            output_dir: Répertoire de sortie pour les fichiers anonymisés

        Returns:
            Dict[str, Any]: Données anonymisées
        """
        # Charger les données extraites
        with open(input_file, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)

        # Anonymiser chaque page
        for page in extracted_data['pages']:
            # Anonymiser le texte nettoyé
            anonymized_text, entities = self._anonymize_text(page['cleaned_text'])
            page['anonymized_text'] = anonymized_text
            page['detected_entities'] = entities

            # Anonymiser les sections
            for section_name, section_text in page['sections'].items():
                anonymized_section, section_entities = self._anonymize_text(section_text)
                page['sections'][section_name] = anonymized_section
                page['detected_entities'].extend(section_entities)

        # Mettre à jour les métadonnées
        extracted_data['metadata']['processing_history'].append({
            "step": "anonymization",
            "details": "Anonymisation avec eds-nlp",
            "entity_count": sum(len(page['detected_entities']) for page in extracted_data['pages'])
        })

        # Sauvegarder le résultat si un répertoire de sortie est spécifié
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{input_file.stem}_anonymized.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=2)

        return extracted_data

    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Dict[str, Any]]:
        """
        Traite tous les fichiers JSON extraits dans un répertoire.

        Args:
            input_dir: Répertoire contenant les fichiers JSON extraits
            output_dir: Répertoire de sortie pour les fichiers anonymisés

        Returns:
            Dict[str, Dict[str, Any]]: Résultats de l'anonymisation
        """
        results = {}
        
        # Traiter tous les fichiers JSON
        for json_file in input_dir.glob("*_extracted.json"):
            try:
                result = self.anonymize_extracted_file(json_file, output_dir)
                results[json_file.stem] = result
            except Exception as e:
                print(f"Erreur lors du traitement de {json_file}: {str(e)}")
                continue

        return results
