import json
import logging
import csv
import concurrent.futures
from pathlib import Path
import edsnlp
import huggingface_hub
from typing import Dict, Any, List, Optional, Tuple
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class MedicalDataAnonymizer:
    """Classe pour anonymiser les données médicales extraites en utilisant eds-pseudo."""
    
    def __init__(self, model: str = "AP-HP/eds-pseudo-public"):
        """Initialise le pipeline NLP avec eds-pseudo.
        
        Args:
            model: Nom du modèle à utiliser (par défaut: "AP-HP/eds-pseudo-public")
        """
        # Récupérer le token depuis les variables d'environnement
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("Le token Hugging Face n'est pas configuré. Créez un fichier .env avec HUGGINGFACE_TOKEN=votre_token")
        
        # Se connecter à Hugging Face et charger le modèle
        huggingface_hub.login(token=token)
        self.nlp = edsnlp.load(model, auto_update=True)
        
        # Liste des entités détectées par eds-pseudo
        self.entity_types = [
            "ADRESSE",    # Adresse postale
            "DATE",       # Date quelconque
            "DATE_NAISSANCE",  # Date de naissance
            "HOPITAL",    # Nom d'hôpital
            "IPP",        # Identifiant patient
            "MAIL",       # Adresse email
            "NDA",        # Numéro de dossier
            "NOM",        # Nom de famille
            "PRENOM",     # Prénom
            "SECU",       # Numéro de sécurité sociale
            "TEL",        # Numéro de téléphone
            "VILLE",      # Nom de ville
            "ZIP"         # Code postal
        ]

    def anonymize_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Anonymise un texte donné et retourne le texte modifié avec les entités détectées.
        
        Args:
            text: Texte à anonymiser
            
        Returns:
            Tuple contenant :
            - Le texte anonymisé
            - La liste des entités détectées avec leurs métadonnées
        """
        # Traiter le texte avec le pipeline
        doc = self.nlp(text)
        anonymized_text = text
        entities = []

        # Collecter toutes les entités détectées (dans l'ordre inverse pour préserver les indices)
        for ent in reversed(doc.ents):
            entity_type = ent.label_
            entity_text = ent.text
            start, end = ent.start_char, ent.end_char
            
            # Créer un identifiant de remplacement descriptif
            replacement = f"[{entity_type}_{len(entities):03d}]"
            
            # Remplacer l'entité dans le texte
            anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
            
            # Sauvegarder les informations sur l'entité
            entities.append({
                "type": entity_type,
                "text": entity_text,
                "start": start,
                "end": end,
                "replacement": replacement,
                "confidence": getattr(ent, "score", None)  # Score de confiance si disponible
            })

        return anonymized_text, entities

    def anonymize_extracted_file(self, input_file: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Anonymise un fichier JSON extrait.
        
        Args:
            input_file: Chemin vers le fichier JSON extrait
            output_dir: Répertoire de sortie pour les fichiers anonymisés
            
        Returns:
            Dict[str, Any]: Données anonymisées avec métadonnées
        """
        # Charger les données extraites
        with open(input_file, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)

        # Anonymiser chaque page
        for page in extracted_data['pages']:
            # Anonymiser le texte principal
            anonymized_text, entities = self.anonymize_text(page['cleaned_text'])
            page['anonymized_text'] = anonymized_text
            page['detected_entities'] = entities

            # Anonymiser les sections si présentes
            if 'sections' in page:
                for section_name, section_text in page['sections'].items():
                    anonymized_section, section_entities = self.anonymize_text(section_text)
                    page['sections'][section_name] = anonymized_section
                    page['detected_entities'].extend(section_entities)

        # Mettre à jour les métadonnées
        if 'metadata' not in extracted_data:
            extracted_data['metadata'] = {}
        if 'processing_history' not in extracted_data['metadata']:
            extracted_data['metadata']['processing_history'] = []
            
        extracted_data['metadata']['processing_history'].append({
            "step": "anonymization",
            "tool": "eds-pseudo",
            "model": self.nlp.meta.get("name", "AP-HP/eds-pseudo-public"),
            "entity_count": sum(len(page['detected_entities']) for page in extracted_data['pages'])
        })

        # Sauvegarder le résultat si un répertoire de sortie est spécifié
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le fichier JSON principal
            output_file = output_dir / f"{input_file.stem}_anonymized.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=2)
            
            # Sauvegarder un tableau de correspondance au format CSV
            table_file = output_dir / f"{input_file.stem}_entities.csv"
            with open(table_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Type", "Texte Original", "Remplacement", "Confiance"])
                for page in extracted_data['pages']:
                    for entity in page['detected_entities']:
                        writer.writerow([
                            entity['type'],
                            entity['text'],
                            entity['replacement'],
                            entity.get('confidence', '')
                        ])

        return extracted_data

    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Traite tous les fichiers JSON extraits dans un répertoire.
        
        Args:
            input_dir: Répertoire contenant les fichiers JSON extraits
            output_dir: Répertoire de sortie pour les fichiers anonymisés
            
        Returns:
            Dict[str, Dict[str, Any]]: Résultats de l'anonymisation par fichier
        """
        results = {}
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Traiter tous les fichiers JSON
        files = list(input_dir.glob("*_extracted.json"))
        total_files = len(files)
        
        logging.info(f"Début de l'anonymisation de {total_files} fichiers...")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(self.anonymize_extracted_file, file, output_dir): file
                for file in files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    results[file.stem] = result
                    logging.info(f"Fichier traité avec succès : {file.name}")
                except Exception as e:
                    logging.error(f"Erreur lors du traitement de {file}: {str(e)}")
        
        logging.info(f"Anonymisation terminée. {len(results)}/{total_files} fichiers traités avec succès.")
        return results
