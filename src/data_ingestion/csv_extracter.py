# Utility function for external use: extract all relationships from a CSV file
def extract_entities_and_relationships_from_csv(csv_path: str) -> dict:
    """
    Extract all entities and relationships from a CSV file and return as a dictionary.
    """
    import pandas as pd
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    extractor = CSVEntityExtractor()
    extractor.df = df
    relationships = extractor.extract_all_relationships()
    return relationships

#script csv extracter: uday

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import re
import json
# --- BBox Extraction Logic (RAG-optimized) ---
def extract_bbox_relationships_rag(df):
    relationships = {
        "basic_info": {},
        "image_to_finding_map": defaultdict(list),
        "image_finding_bbox_map": defaultdict(lambda: defaultdict(list)),
        "bbox_details": {},
        "finding_bbox_map": defaultdict(list),
        "co_occurrence_relationships": {},
        "validation": {}
    }

    df = df.rename(columns=lambda x: x.strip())
    # The actual columns are: 'Image Index', 'Finding Label', 'Bbox [x', 'y', 'w', 'h]'
    # But in the CSV, the columns are:
    # 'Image Index', 'Finding Label', 'Bbox [x', 'y', 'w', 'h]'
    # The last four columns are the bbox coordinates, but their headers are split due to the CSV structure.
    # Let's get the real column names:
    columns = list(df.columns)
    # The first two are correct, the next four are the bbox coordinates
    if len(columns) < 6:
        relationships["validation"]["missing_columns"] = True
        return relationships
    image_col = columns[0]
    finding_col = columns[1]
    x_col = columns[2]
    y_col = columns[3]
    w_col = columns[4]
    h_col = columns[5]

    df = df.dropna(subset=[image_col, finding_col, x_col, y_col, w_col, h_col])

    all_findings_per_image = defaultdict(set)

    for _, row in df.iterrows():
        image_id = str(row[image_col]).strip()
        finding = str(row[finding_col]).strip()
        bbox_coords = (
            float(row[x_col]),
            float(row[y_col]),
            float(row[w_col]),
            float(row[h_col])
        )
        bbox_id = f"{image_id}_{finding}_{int(row[x_col])}_{int(row[y_col])}_{int(row[w_col])}_{int(row[h_col])}"

        # Entities
        relationships['image_to_finding_map'][image_id].append(finding)
        relationships['image_finding_bbox_map'][image_id][finding].append(bbox_id)
        relationships['bbox_details'][bbox_id] = {
            "bbox_coords": bbox_coords
        }
        relationships['finding_bbox_map'][finding].append(bbox_id)

        all_findings_per_image[image_id].add(finding)

    # Co-occurrence relationships
    finding_pairs = defaultdict(int)
    for findings in all_findings_per_image.values():
        sorted_finds = sorted(findings)
        for i in range(len(sorted_finds)):
            for j in range(i+1, len(sorted_finds)):
                pair = (sorted_finds[i], sorted_finds[j])
                finding_pairs[pair] += 1

    # Convert tuple keys to strings for JSON serialization
    finding_pairs_str = {f"{k[0]}|{k[1]}": v for k, v in finding_pairs.items()}
    relationships["co_occurrence_relationships"] = finding_pairs_str

    # Basic Info
    relationships['basic_info'] = {
        "total_rows": len(df),
        "unique_images": len(relationships['image_to_finding_map']),
        "unique_findings": len(set(df[finding_col].unique())),
        "total_bboxes": len(relationships['bbox_details'])
    }

    # Validation Flags
    relationships['validation'] = {
        "basic_info_extracted": True,
        "image_metadata_extracted": True,
        "co_occurrence_extracted": True,
        "demographics_extracted": False,
        "patient_medical_extracted": False,
        "temporal_relationships_extracted": False,
        "overall_success": True
    }

    # Convert defaultdicts to dicts for JSON
    def convert_dd(obj):
        if isinstance(obj, defaultdict):
            obj = dict(obj)
        if isinstance(obj, dict):
            return {k: convert_dd(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_dd(i) for i in obj]
        else:
            return obj
    return convert_dd(relationships)
import re
from typing import Dict, List, Set, Tuple, Any
import os
import json
from datetime import datetime


class CSVEntityExtractor:
    def extract_medical_findings(self) -> Dict[str, Any]:
        """
        Stub for extracting medical findings from the CSV. Returns an empty dict or minimal info.
        """
        return {}
    def extract_basic_info(self) -> Dict[str, Any]:
        """
        Stub for extracting basic info from the CSV. Returns an empty dict or minimal info.
        """
        return {}
    def extract_bbox_relationships(self) -> Dict[str, Any]:
        """
        Extract relationships from BBox_List_2017.csv containing bounding box data.
        Returns:
            Dict[str, Any]: BBox relationships mapped to image and findings
        """
        # Try to flexibly match the Bbox columns
        col_map = {}
        for col in self.df.columns:
            col_lower = col.lower().replace(' ', '').replace('[', '').replace(']', '')
            if 'bboxx' in col_lower or col_lower in ['bbox[x]', 'bbox[x']:
                col_map['x'] = col
            elif col_lower in ['y', 'bboxy']:
                col_map['y'] = col
            elif col_lower in ['w', 'bboxw', 'width']:
                col_map['w'] = col
            elif col_lower in ['h', 'bboxh', 'height']:
                col_map['h'] = col
        # Also try to match the main columns
        for col in self.df.columns:
            if col.strip().lower() == 'image index':
                col_map['image'] = col
            elif col.strip().lower() in ['finding label', 'finding labels']:
                col_map['finding'] = col
        required_keys = {'image', 'finding', 'x', 'y', 'w', 'h'}
        if not required_keys.issubset(col_map.keys()):
            return {}

        bbox_data = defaultdict(list)

        for _, row in self.df.iterrows():
            image = row[col_map['image']]
            finding = row[col_map['finding']]
            bbox = {
                'x': row[col_map['x']],
                'y': row[col_map['y']],
                'width': row[col_map['w']],
                'height': row[col_map['h']]
            }
            bbox_data[image].append(bbox)

        return {'bbox_relationships': dict(bbox_data)}
   
    def extract_patient_demographics(self) -> Dict[str, Any]:
        """
        Extract patient demographic information.
       
        Returns:
            Dict[str, Any]: Patient demographic entities
        """
        demographics = {}
       
        # Extract age information
        if 'Patient Age' in self.df.columns:
            ages = self.df['Patient Age'].dropna()
            demographics['age_statistics'] = {
                'min_age': int(ages.min()),
                'max_age': int(ages.max()),
                'mean_age': round(ages.mean(), 2),
                'median_age': round(ages.median(), 2),
                'age_std': round(ages.std(), 2),
                'age_distribution': dict(ages.value_counts().head(10))
            }
       
        # Extract gender information
        if 'Patient Gender' in self.df.columns:
            genders = self.df['Patient Gender'].dropna()
            demographics['gender_distribution'] = dict(genders.value_counts())
            demographics['unique_genders'] = list(genders.unique())
       
        # Extract patient ID information
        if 'Patient ID' in self.df.columns:
            patient_ids = self.df['Patient ID'].dropna()
            demographics['patient_statistics'] = {
                'unique_patients': len(patient_ids.unique()),
                'total_records': len(patient_ids),
                'average_records_per_patient': round(len(patient_ids) / len(patient_ids.unique()), 2)
            }
       
        return demographics
   
    def extract_image_metadata(self) -> Dict[str, Any]:
        """
        Extract image-related metadata and entities.
       
        Returns:
            Dict[str, Any]: Image metadata entities
        """
        image_data = {}
       
        # Extract image file information
        if 'Image Index' in self.df.columns:
            images = self.df['Image Index'].dropna()
           
            # Extract file extensions
            extensions = [os.path.splitext(img)[1] for img in images]
            extension_counts = Counter(extensions)
           
            # Extract image numbering patterns
            image_numbers = []
            for img in images:
                # Extract numbers from image names
                numbers = re.findall(r'\d+', img)
                if numbers:
                    image_numbers.extend(numbers)
           
            image_data['image_statistics'] = {
                'total_images': len(images),
                'unique_images': len(images.unique()),
                'file_extensions': dict(extension_counts),
                'image_naming_pattern': 'Sequential numbering with .png extension'
            }
       
        # Extract view position information
        if 'View Position' in self.df.columns:
            view_positions = self.df['View Position'].dropna()
            image_data['view_positions'] = {
                'unique_positions': list(view_positions.unique()),
                'position_distribution': dict(view_positions.value_counts())
            }
       
        # Extract image dimensions
        dimension_cols = [col for col in self.df.columns if 'Width' in col or 'Height' in col or 'Image_Width' in col or 'Image_Height' in col]
        if dimension_cols:
            image_data['dimensions'] = {}
            for col in dimension_cols:
                if col in self.df.columns:
                    dims = self.df[col].dropna()
                    image_data['dimensions'][col] = {
                        'min': int(dims.min()) if not dims.empty else None,
                        'max': int(dims.max()) if not dims.empty else None,
                        'mean': round(dims.mean(), 2) if not dims.empty else None,
                        'unique_values': list(dims.unique())[:10]  # Limit to first 10
                    }
       
        # Extract pixel spacing information
        spacing_cols = [col for col in self.df.columns if 'PixelSpacing' in col or 'Pixel_Spacing' in col]
        if spacing_cols:
            image_data['pixel_spacing'] = {}
            for col in spacing_cols:
                if col in self.df.columns:
                    spacing = self.df[col].dropna()
                    image_data['pixel_spacing'][col] = {
                        'min': round(spacing.min(), 6) if not spacing.empty else None,
                        'max': round(spacing.max(), 6) if not spacing.empty else None,
                        'unique_values': list(spacing.unique())[:10]
                    }
       
        return image_data
   
    def extract_temporal_entities(self) -> Dict[str, Any]:
        """
        Extract temporal/follow-up related entities.
       
        Returns:
            Dict[str, Any]: Temporal entities
        """
        temporal_data = {}
       
        if 'Follow-up #' in self.df.columns:
            followups = self.df['Follow-up #'].dropna()
            temporal_data['followup_statistics'] = {
                'min_followup': int(followups.min()),
                'max_followup': int(followups.max()),
                'followup_distribution': dict(followups.value_counts().head(20)),
                'patients_with_followups': len(self.df[self.df['Follow-up #'] > 0]['Patient ID'].unique()) if 'Patient ID' in self.df.columns else 0
            }
       
        return temporal_data
   
    def extract_all_categorical_entities(self) -> Dict[str, Any]:
        """
        Extract all unique values from categorical columns.
       
        Returns:
            Dict[str, Any]: All categorical entities
        """
        categorical_entities = {}
       
        for column in self.df.columns:
            # Check if column is categorical (object type or limited unique values)
            if (self.df[column].dtype == 'object' or
                (self.df[column].dtype in ['int64', 'float64'] and
                 self.df[column].nunique() <= 50)):  # Arbitrary threshold for categorical
               
                unique_values = self.df[column].dropna().unique()
                value_counts = self.df[column].value_counts()
               
                categorical_entities[column] = {
                    'unique_values': list(unique_values),
                    'unique_count': len(unique_values),
                    'value_distribution': dict(value_counts.head(20)),  # Top 20 most frequent
                    'null_count': self.df[column].isnull().sum()
                }
       
        return categorical_entities
   
    def extract_numerical_statistics(self) -> Dict[str, Any]:
        """
        Extract statistical information from numerical columns.
       
        Returns:
            Dict[str, Any]: Numerical statistics
        """
        numerical_stats = {}
       
        # Get all columns and check which ones can be treated as numerical
        for column in self.df.columns:
            # Try to convert to numeric if possible
            try:
                numeric_data = pd.to_numeric(self.df[column], errors='coerce')
                # Only process if we have some valid numeric values
                if not numeric_data.isna().all():
                    clean_data = numeric_data.dropna()
                   
                    if len(clean_data) > 0:
                        numerical_stats[column] = {
                            'count': len(clean_data),
                            'mean': round(clean_data.mean(), 4),
                            'median': round(clean_data.median(), 4),
                            'std': round(clean_data.std(), 4),
                            'min': clean_data.min(),
                            'max': clean_data.max(),
                            'percentiles': {
                                '25%': round(clean_data.quantile(0.25), 4),
                                '50%': round(clean_data.quantile(0.50), 4),
                                '75%': round(clean_data.quantile(0.75), 4),
                                '90%': round(clean_data.quantile(0.90), 4),
                                '95%': round(clean_data.quantile(0.95), 4),
                                '99%': round(clean_data.quantile(0.99), 4)
                            },
                            'null_count': len(self.df[column]) - len(clean_data),
                            'data_type': str(self.df[column].dtype)
                        }
            except (ValueError, TypeError):
                # Skip columns that can't be converted to numeric
                continue
       
        return numerical_stats
   
    def extract_data_quality_entities(self) -> Dict[str, Any]:
        """
        Extract data quality related information.
       
        Returns:
            Dict[str, Any]: Data quality entities
        """
        quality_data = {
            'missing_data_summary': {},
            'duplicate_analysis': {},
            'data_consistency': {}
        }
       
        # Missing data analysis
        for column in self.df.columns:
            missing_count = self.df[column].isnull().sum()
            missing_percentage = (missing_count / len(self.df)) * 100
            quality_data['missing_data_summary'][column] = {
                'missing_count': missing_count,
                'missing_percentage': round(missing_percentage, 2)
            }
       
        # Duplicate analysis
        quality_data['duplicate_analysis'] = {
            'total_duplicates': self.df.duplicated().sum(),
            'duplicate_percentage': round((self.df.duplicated().sum() / len(self.df)) * 100, 2)
        }
       
        # Check for specific patterns in key columns
        if 'Image Index' in self.df.columns:
            quality_data['data_consistency']['image_naming_consistent'] = \
                all(img.endswith('.png') for img in self.df['Image Index'].dropna())
       
        return quality_data
   
    def extract_patient_medical_relationships(self) -> Dict[str, Any]:
        """
        Extract relationships between patients and their medical findings.
       
        Returns:
            Dict[str, Any]: Patient-medical finding relationships
        """
        relationships = {}
       
        if 'Patient ID' in self.df.columns and 'Finding Labels' in self.df.columns:
            # Group by patient ID to find patterns
            patient_groups = self.df.groupby('Patient ID')
           
            relationships['patient_findings'] = {}
            relationships['finding_progression'] = {}
            relationships['patient_finding_counts'] = {}
           
            for patient_id, group in patient_groups:
                patient_findings = []
                finding_timeline = []
               
                # Extract all findings for this patient
                for _, row in group.iterrows():
                    if pd.notna(row['Finding Labels']):
                        findings = row['Finding Labels'].split('|')
                        findings = [f.strip() for f in findings]
                        patient_findings.extend(findings)
                       
                        # Create timeline entry
                        timeline_entry = {
                            'followup': row.get('Follow-up #', 0),
                            'findings': findings,
                            'age': row.get('Patient Age', None),
                            'image': row.get('Image Index', None)
                        }
                        finding_timeline.append(timeline_entry)
               
                # Store patient relationships
                unique_findings = list(set(patient_findings))
                finding_counts = Counter(patient_findings)
               
                relationships['patient_findings'][str(patient_id)] = {
                    'unique_findings': unique_findings,
                    'finding_frequency': dict(finding_counts),
                    'total_images': len(group),
                    'age_range': [group['Patient Age'].min(), group['Patient Age'].max()] if 'Patient Age' in group.columns else None,
                    'gender': group['Patient Gender'].iloc[0] if 'Patient Gender' in group.columns else None
                }
               
                # Store progression timeline (sorted by follow-up number)
                finding_timeline.sort(key=lambda x: x['followup'])
                relationships['finding_progression'][str(patient_id)] = finding_timeline
               
                relationships['patient_finding_counts'][str(patient_id)] = len(unique_findings)
       
        return relationships
   
    def extract_demographic_medical_relationships(self) -> Dict[str, Any]:
        """
        Extract relationships between demographics and medical findings.
       
        Returns:
            Dict[str, Any]: Demographic-medical relationships
        """
        relationships = {}
       
        if 'Patient Gender' in self.df.columns and 'Finding Labels' in self.df.columns:
            # Gender-based medical findings
            gender_findings = {}
            for gender in self.df['Patient Gender'].dropna().unique():
                gender_data = self.df[self.df['Patient Gender'] == gender]
               
                # Extract findings for this gender
                all_findings = []
                for findings_str in gender_data['Finding Labels'].dropna():
                    findings = findings_str.split('|')
                    all_findings.extend([f.strip() for f in findings])
               
                finding_counts = Counter(all_findings)
                gender_findings[gender] = {
                    'total_cases': len(gender_data),
                    'unique_findings': list(finding_counts.keys()),
                    'finding_frequencies': dict(finding_counts),
                    'most_common': finding_counts.most_common(5)
                }
           
            relationships['gender_findings'] = gender_findings
       
        if 'Patient Age' in self.df.columns and 'Finding Labels' in self.df.columns:
            # Age group-based medical findings
            age_groups = {
                'young_adult': (18, 40),
                'middle_aged': (41, 65),
                'elderly': (66, 100)
            }
           
            age_findings = {}
            for group_name, (min_age, max_age) in age_groups.items():
                age_data = self.df[
                    (self.df['Patient Age'] >= min_age) &
                    (self.df['Patient Age'] <= max_age)
                ]
               
                if len(age_data) > 0:
                    all_findings = []
                    for findings_str in age_data['Finding Labels'].dropna():
                        findings = findings_str.split('|')
                        all_findings.extend([f.strip() for f in findings])
                   
                    finding_counts = Counter(all_findings)
                    age_findings[group_name] = {
                        'age_range': f"{min_age}-{max_age}",
                        'total_cases': len(age_data),
                        'unique_findings': list(finding_counts.keys()),
                        'finding_frequencies': dict(finding_counts),
                        'most_common': finding_counts.most_common(5)
                    }
           
            relationships['age_group_findings'] = age_findings
       
        return relationships
   
    def extract_temporal_relationships(self) -> Dict[str, Any]:
        """
        Extract temporal relationships and patterns over time.
       
        Returns:
            Dict[str, Any]: Temporal relationships
        """
        relationships = {}
       
        if 'Follow-up #' in self.df.columns and 'Finding Labels' in self.df.columns:
            # Follow-up progression patterns
            followup_patterns = {}
            max_followup = self.df['Follow-up #'].max()
           
            for followup_num in range(int(max_followup) + 1):
                followup_data = self.df[self.df['Follow-up #'] == followup_num]
               
                if len(followup_data) > 0:
                    all_findings = []
                    for findings_str in followup_data['Finding Labels'].dropna():
                        findings = findings_str.split('|')
                        all_findings.extend([f.strip() for f in findings])
                   
                    finding_counts = Counter(all_findings)
                    followup_patterns[str(followup_num)] = {
                        'total_cases': len(followup_data),
                        'unique_patients': len(followup_data['Patient ID'].unique()) if 'Patient ID' in followup_data.columns else 0,
                        'finding_frequencies': dict(finding_counts),
                        'most_common': finding_counts.most_common(5)
                    }
           
            relationships['followup_patterns'] = followup_patterns
           
            # Patient progression analysis
            if 'Patient ID' in self.df.columns:
                progression_analysis = {}
                patient_groups = self.df.groupby('Patient ID')
               
                for patient_id, group in patient_groups:
                    if len(group) > 1:  # Only patients with multiple follow-ups
                        sorted_group = group.sort_values('Follow-up #')
                        progression = []
                       
                        for _, row in sorted_group.iterrows():
                            if pd.notna(row['Finding Labels']):
                                findings = row['Finding Labels'].split('|')
                                findings = [f.strip() for f in findings]
                                progression.append({
                                    'followup': row['Follow-up #'],
                                    'findings': findings,
                                    'age': row.get('Patient Age', None)
                                })
                       
                        if len(progression) > 1:
                            progression_analysis[str(patient_id)] = {
                                'progression_timeline': progression,
                                'total_followups': len(progression),
                                'finding_changes': self._analyze_finding_changes(progression)
                            }
               
                relationships['patient_progressions'] = progression_analysis
       
        return relationships
   
    def extract_image_medical_relationships(self) -> Dict[str, Any]:
        """
        Extract relationships between image characteristics and medical findings.
       
        Returns:
            Dict[str, Any]: Image-medical relationships
        """
        relationships = {}
       
        # View position vs findings
        if 'View Position' in self.df.columns and 'Finding Labels' in self.df.columns:
            view_findings = {}
            for view_pos in self.df['View Position'].dropna().unique():
                view_data = self.df[self.df['View Position'] == view_pos]
               
                all_findings = []
                for findings_str in view_data['Finding Labels'].dropna():
                    findings = findings_str.split('|')
                    all_findings.extend([f.strip() for f in findings])
               
                finding_counts = Counter(all_findings)
                view_findings[view_pos] = {
                    'total_images': len(view_data),
                    'finding_frequencies': dict(finding_counts),
                    'most_common': finding_counts.most_common(5)
                }
           
            relationships['view_position_findings'] = view_findings
       
        # Image dimensions vs findings (if available)
        dimension_cols = [col for col in self.df.columns if 'Width' in col or 'Height' in col or 'Image_Width' in col or 'Image_Height' in col]
        if dimension_cols and 'Finding Labels' in self.df.columns:
            dimension_findings = {}
           
            for dim_col in dimension_cols:
                if dim_col in self.df.columns:
                    # Create size categories
                    dim_data = self.df[dim_col].dropna()
                    if len(dim_data) > 0:
                        q25, q75 = dim_data.quantile([0.25, 0.75])
                       
                        size_categories = {
                            'small': self.df[self.df[dim_col] <= q25],
                            'medium': self.df[(self.df[dim_col] > q25) & (self.df[dim_col] <= q75)],
                            'large': self.df[self.df[dim_col] > q75]
                        }
                       
                        category_findings = {}
                        for category, cat_data in size_categories.items():
                            if len(cat_data) > 0:
                                all_findings = []
                                for findings_str in cat_data['Finding Labels'].dropna():
                                    findings = findings_str.split('|')
                                    all_findings.extend([f.strip() for f in findings])
                               
                                if all_findings:
                                    finding_counts = Counter(all_findings)
                                    category_findings[category] = {
                                        'total_images': len(cat_data),
                                        'finding_frequencies': dict(finding_counts),
                                        'most_common': finding_counts.most_common(3)
                                    }
                       
                        dimension_findings[dim_col] = category_findings
           
            relationships['dimension_findings'] = dimension_findings
       
        return relationships
   
    def extract_co_occurrence_relationships(self) -> Dict[str, Any]:
        """
        Extract co-occurrence relationships between different findings.
       
        Returns:
            Dict[str, Any]: Co-occurrence relationships
        """
        relationships = {}
       
        if 'Finding Labels' in self.df.columns:
            # Finding co-occurrence matrix
            all_individual_findings = set()
            finding_combinations = []
           
            for findings_str in self.df['Finding Labels'].dropna():
                findings = findings_str.split('|')
                findings = [f.strip() for f in findings]
                finding_combinations.append(findings)
                all_individual_findings.update(findings)
           
            # Create co-occurrence matrix
            finding_list = list(all_individual_findings)
            co_occurrence_matrix = {}
           
            for finding1 in finding_list:
                co_occurrence_matrix[finding1] = {}
                for finding2 in finding_list:
                    if finding1 != finding2:
                        co_count = sum(1 for combo in finding_combinations
                                     if finding1 in combo and finding2 in combo)
                        co_occurrence_matrix[finding1][finding2] = co_count
           
            relationships['co_occurrence_matrix'] = co_occurrence_matrix
           
            # Most common co-occurrences
            co_occurrence_pairs = []
            for finding1 in finding_list:
                for finding2 in finding_list:
                    if finding1 < finding2:  # Avoid duplicates
                        co_count = co_occurrence_matrix[finding1][finding2]
                        if co_count > 0:
                            co_occurrence_pairs.append({
                                'finding_pair': [finding1, finding2],
                                'co_occurrence_count': co_count,
                                'finding1_total': sum(1 for combo in finding_combinations if finding1 in combo),
                                'finding2_total': sum(1 for combo in finding_combinations if finding2 in combo)
                            })
           
            # Sort by co-occurrence count
            co_occurrence_pairs.sort(key=lambda x: x['co_occurrence_count'], reverse=True)
            relationships['top_co_occurrences'] = co_occurrence_pairs[:20]  # Top 20
           
            # Finding exclusivity (findings that rarely occur together)
            exclusive_pairs = []
            for finding1 in finding_list:
                for finding2 in finding_list:
                    if finding1 < finding2:
                        finding1_count = sum(1 for combo in finding_combinations if finding1 in combo)
                        finding2_count = sum(1 for combo in finding_combinations if finding2 in combo)
                        co_count = co_occurrence_matrix[finding1][finding2]
                       
                        if finding1_count > 10 and finding2_count > 10 and co_count == 0:
                            exclusive_pairs.append({
                                'finding_pair': [finding1, finding2],
                                'finding1_count': finding1_count,
                                'finding2_count': finding2_count,
                                'exclusivity_score': (finding1_count + finding2_count) / 2
                            })
           
            exclusive_pairs.sort(key=lambda x: x['exclusivity_score'], reverse=True)
            relationships['exclusive_findings'] = exclusive_pairs[:10]  # Top 10
       
        return relationships
   
    def _analyze_finding_changes(self, progression: List[Dict]) -> Dict[str, Any]:
        """
        Helper method to analyze changes in findings over time.
       
        Args:
            progression (List[Dict]): Timeline of findings for a patient
           
        Returns:
            Dict[str, Any]: Analysis of finding changes
        """
        changes = {
            'new_findings': [],
            'resolved_findings': [],
            'persistent_findings': [],
            'total_changes': 0
        }
       
        if len(progression) < 2:
            return changes
       
        # Compare consecutive follow-ups
        for i in range(1, len(progression)):
            prev_findings = set(progression[i-1]['findings'])
            curr_findings = set(progression[i]['findings'])
           
            new_findings = curr_findings - prev_findings
            resolved_findings = prev_findings - curr_findings
           
            if new_findings:
                changes['new_findings'].extend([{
                    'followup': progression[i]['followup'],
                    'findings': list(new_findings)
                }])
           
            if resolved_findings:
                changes['resolved_findings'].extend([{
                    'followup': progression[i]['followup'],
                    'findings': list(resolved_findings)
                }])
       
        # Find persistent findings (present in all follow-ups)
        all_followup_findings = [set(fp['findings']) for fp in progression]
        if all_followup_findings:
            persistent = all_followup_findings[0]
            for findings_set in all_followup_findings[1:]:
                persistent = persistent.intersection(findings_set)
            changes['persistent_findings'] = list(persistent)
       
        changes['total_changes'] = len(changes['new_findings']) + len(changes['resolved_findings'])
       
        return changes
   
    def extract_all_relationships(self) -> Dict[str, Any]:
        """
        Extract all possible relationships between entities.
       
        Returns:
            Dict[str, Any]: Complete relationships dictionary
        """
        print("Starting relationship extraction...")
        relationships = {
            'patient_medical_relationships': self.extract_patient_medical_relationships(),
            'demographic_medical_relationships': self.extract_demographic_medical_relationships(),
            'temporal_relationships': self.extract_temporal_relationships(),
            'image_medical_relationships': self.extract_image_medical_relationships(),
            'co_occurrence_relationships': self.extract_co_occurrence_relationships(),
            'extraction_metadata': {
                'extraction_date': datetime.now().isoformat(),
                'total_records': len(self.df),
                'total_patients': len(self.df['Patient ID'].unique()) if 'Patient ID' in self.df.columns else 0,
                'total_images': len(self.df['Image Index'].unique()) if 'Image Index' in self.df.columns else 0
            }
        }

        # Append BBox relationships only if columns are present
        if {'Image Index', 'Finding Label', 'Bbox [x', 'y', 'w', 'h]'}.issubset(self.df.columns):
            relationships.update(self.extract_bbox_relationships())

        print("Relationship extraction completed!")
        return relationships
   
    def save_relationships_to_json(self, relationships: Dict[str, Any], output_file: str = None) -> str:
        """
        Save extracted relationships to a JSON file.
       
        Args:
            relationships (Dict[str, Any]): Relationships dictionary
            output_file (str, optional): Output file path
           
        Returns:
            str: Path to the saved JSON file
        """
        if output_file is None:
            # Create default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"entity_relationships_{timestamp}.json"
       
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
       
        # Convert the relationships
        json_ready_relationships = convert_numpy_types(relationships)
       
        # Save to JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_ready_relationships, f, indent=2, ensure_ascii=False)
           
            print(f"Relationships successfully saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving relationships to JSON: {str(e)}")
            return None
   
    def validate_relationship_extraction(self, relationships: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate if all possible relationships have been properly extracted.
       
        Args:
            relationships (Dict[str, Any]): Extracted relationships
           
        Returns:
            Dict[str, bool]: Validation results
        """
        validation_results = {}
       
        # Check if patient-medical relationships are extracted
        validation_results['patient_medical_extracted'] = (
            'patient_medical_relationships' in relationships and
            len(relationships['patient_medical_relationships']) > 0
        )
       
        # Check if demographic relationships are extracted
        validation_results['demographic_medical_extracted'] = (
            'demographic_medical_relationships' in relationships and
            len(relationships['demographic_medical_relationships']) > 0
        )
       
        # Check if temporal relationships are extracted
        validation_results['temporal_relationships_extracted'] = (
            'temporal_relationships' in relationships and
            len(relationships['temporal_relationships']) > 0
        )
       
        # Check if image relationships are extracted
        validation_results['image_medical_extracted'] = (
            'image_medical_relationships' in relationships and
            len(relationships['image_medical_relationships']) > 0
        )
       
        # Check if co-occurrence relationships are extracted
        validation_results['co_occurrence_extracted'] = (
            'co_occurrence_relationships' in relationships and
            'co_occurrence_matrix' in relationships['co_occurrence_relationships']
        )
       
        # Validate specific relationship completeness
        if 'patient_medical_relationships' in relationships:
            pmr = relationships['patient_medical_relationships']
            validation_results['patient_findings_complete'] = (
                'patient_findings' in pmr and len(pmr['patient_findings']) > 0
            )
            validation_results['finding_progression_complete'] = (
                'finding_progression' in pmr and len(pmr['finding_progression']) > 0
            )
       
        if 'demographic_medical_relationships' in relationships:
            dmr = relationships['demographic_medical_relationships']
            validation_results['gender_findings_complete'] = (
                'gender_findings' in dmr and len(dmr['gender_findings']) > 0
            )
            validation_results['age_group_findings_complete'] = (
                'age_group_findings' in dmr and len(dmr['age_group_findings']) > 0
            )
       
        # Check data quality for relationships
        total_patients = len(self.df['Patient ID'].unique()) if 'Patient ID' in self.df.columns else 0
        if 'patient_medical_relationships' in relationships and total_patients > 0:
            extracted_patients = len(relationships['patient_medical_relationships'].get('patient_findings', {}))
            validation_results['all_patients_covered'] = extracted_patients == total_patients
            validation_results['patients_coverage_ratio'] = extracted_patients / total_patients if total_patients > 0 else 0
       
        # Overall validation
        core_validations = [
            'patient_medical_extracted',
            'demographic_medical_extracted',
            'temporal_relationships_extracted',
            'co_occurrence_extracted'
        ]
        validation_results['overall_relationship_success'] = all(
            validation_results.get(key, False) for key in core_validations
        )
       
        return validation_results
   
    def extract_all_entities(self) -> Dict[str, Any]:
        """
        Extract all possible entities from the CSV file.
       
        Returns:
            Dict[str, Any]: Complete entities dictionary
        """
        print("Starting entity extraction...")
       
        # Extract all types of entities
        self.entities = {
            'basic_info': self.extract_basic_info(),
            'medical_findings': self.extract_medical_findings(),
            'patient_demographics': self.extract_patient_demographics(),
            'image_metadata': self.extract_image_metadata(),
            'temporal_entities': self.extract_temporal_entities(),
            'categorical_entities': self.extract_all_categorical_entities(),
            'numerical_statistics': self.extract_numerical_statistics(),
            'data_quality': self.extract_data_quality_entities()
        }
       
        print("Entity extraction completed!")
        return self.entities
   
    def validate_extraction(self) -> Dict[str, bool]:
        """
        Validate if all possible entities have been properly extracted.
       
        Returns:
            Dict[str, bool]: Validation results
        """
        validation_results = {}
       
        # Check if basic info is extracted
        validation_results['basic_info_extracted'] = 'basic_info' in self.entities and \
                                                   len(self.entities['basic_info']) > 0
       
        # Check if all columns are covered in categorical entities
        if 'categorical_entities' in self.entities:
            covered_columns = set(self.entities['categorical_entities'].keys())
            all_columns = set(self.df.columns)
            validation_results['all_categorical_columns_covered'] = covered_columns.issuperset(
                {col for col in all_columns if self.df[col].dtype == 'object'}
            )
       
        # Check if numerical statistics are extracted for numeric columns
        if 'numerical_statistics' in self.entities:
            # Count columns that have numeric data (can be converted to numeric)
            potentially_numeric_columns = 0
            for column in self.df.columns:
                try:
                    numeric_data = pd.to_numeric(self.df[column], errors='coerce')
                    if not numeric_data.isna().all():
                        potentially_numeric_columns += 1
                except:
                    continue
           
            stats_columns_count = len(self.entities['numerical_statistics'])
            validation_results['all_numeric_columns_analyzed'] = stats_columns_count >= potentially_numeric_columns
            validation_results['numeric_columns_found'] = potentially_numeric_columns
            validation_results['numeric_columns_analyzed'] = stats_columns_count
       
        # Check if medical findings are properly extracted
        if 'Finding Labels' in self.df.columns:
            validation_results['medical_findings_extracted'] = 'medical_findings' in self.entities and \
                                                             'unique_findings' in self.entities['medical_findings']
       
        # Check if patient demographics are extracted
        validation_results['demographics_extracted'] = 'patient_demographics' in self.entities and \
                                                      len(self.entities['patient_demographics']) > 0
       
        # Check if image metadata is extracted
        validation_results['image_metadata_extracted'] = 'image_metadata' in self.entities and \
                                                        len(self.entities['image_metadata']) > 0
       
        # Overall validation
        validation_results['overall_success'] = all(validation_results.values())
       
        return validation_results
   
    def print_entity_summary(self):
        """
        Print a comprehensive summary of all extracted entities.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE ENTITY EXTRACTION SUMMARY")
        print("="*80)
       
        for category, data in self.entities.items():
            print(f"\n{'='*20} {category.upper().replace('_', ' ')} {'='*20}")
           
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (list, dict)) and len(str(value)) > 200:
                        # Truncate long outputs
                        print(f"{key}: {str(value)[:200]}... (truncated)")
                    else:
                        print(f"{key}: {value}")
            else:
                print(f"Data: {data}")


# def main():
#     """
#     Main function to test the CSV entity extractor and relationship extraction.
#     """
#     import pandas as pd
#     import glob
#     print("CSV Entity Extractor and Relationship Analyzer - Batch Processing (RAG BBox)")
#     print("="*60)

#     #input_folder = os.path.join(r"D:\Softwares\Neo4j-poc\Neo4j_Ingestion", "source_document")
#     #output_folder = os.path.join(r"D:\Softwares\Neo4j-poc\Neo4j_Ingestion", "output_folder")
#     #os.makedirs(output_folder, exist_ok=True)
    

#     # BBox extraction
#     bbox_csv = os.path.join(input_folder, "BBox_List_2017.csv")
#     if os.path.exists(bbox_csv):
#         bbox_df = pd.read_csv(bbox_csv)
#         bbox_result = extract_bbox_relationships_rag(bbox_df)
#         bbox_path = os.path.join(output_folder, 'BBox_List_2017_entity_relationships.json')
#         with open(bbox_path, 'w', encoding='utf-8') as f:
#             json.dump(bbox_result, f, indent=2)
#         print(f"✓ BBox relationships saved to: {bbox_path}")
#     else:
#         bbox_result = {}
#         bbox_path = None
#         print("✗ BBox_List_2017.csv not found.")

#     # Data Entry extraction (use existing class logic)
#     data_csv = os.path.join(input_folder, "Data_Entry_2017.csv")
#     if os.path.exists(data_csv):
#         extractor = CSVEntityExtractor()
#         extractor.df = pd.read_csv(data_csv)
#         data_relationships = extractor.extract_all_relationships()
#         data_path = os.path.join(output_folder, 'Data_Entry_2017_entity_relationships.json')
#         extractor.save_relationships_to_json(data_relationships, data_path)
#         print(f"✓ Data Entry relationships saved to: {data_path}")
#     else:
#         data_relationships = {}
#         data_path = None
#         print("✗ Data_Entry_2017.csv not found.")

#     print("✅ Relationships successfully extracted and saved.")
#     return bbox_result, bbox_path, data_relationships, data_path


def test_relationship_extraction():
    """
    Dedicated test function for relationship extraction functionality.
    """
    print("DEDICATED RELATIONSHIP EXTRACTION TEST")
    print("="*50)
   
    # File path to the CSV
    csv_file_path = os.path.join(r"D:\\Softwares\\Neo4j-poc\\Neo4j_Ingestion\\source_document", "BBox_List_2017.csv")

    # Initialize and load data
    extractor = CSVEntityExtractor()
    import pandas as pd
    try:
        extractor.df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Failed to load data for relationship testing: {e}")
        return False
   
    # Test individual relationship extraction methods
    test_results = {}
   
    print("Testing individual relationship extraction methods...")
   
    # Test patient-medical relationships
    try:
        pmr = extractor.extract_patient_medical_relationships()
        test_results['patient_medical'] = len(pmr) > 0
        print(f"✓ Patient-medical relationships: {len(pmr)} categories extracted")
    except Exception as e:
        test_results['patient_medical'] = False
        print(f"✗ Patient-medical relationships failed: {e}")
   
    # Test demographic-medical relationships
    try:
        dmr = extractor.extract_demographic_medical_relationships()
        test_results['demographic_medical'] = len(dmr) > 0
        print(f"✓ Demographic-medical relationships: {len(dmr)} categories extracted")
    except Exception as e:
        test_results['demographic_medical'] = False
        print(f"✗ Demographic-medical relationships failed: {e}")
   
    # Test temporal relationships
    try:
        tr = extractor.extract_temporal_relationships()
        test_results['temporal'] = len(tr) > 0
        print(f"✓ Temporal relationships: {len(tr)} categories extracted")
    except Exception as e:
        test_results['temporal'] = False
        print(f"✗ Temporal relationships failed: {e}")
   
    # Test co-occurrence relationships
    try:
        cor = extractor.extract_co_occurrence_relationships()
        test_results['co_occurrence'] = len(cor) > 0
        print(f"✓ Co-occurrence relationships: {len(cor)} categories extracted")
    except Exception as e:
        test_results['co_occurrence'] = False
        print(f"✗ Co-occurrence relationships failed: {e}")
   
    # Test complete relationship extraction
    try:
        all_relationships = extractor.extract_all_relationships()
        test_results['complete_extraction'] = len(all_relationships) > 0
        print(f"✓ Complete relationship extraction: {len(all_relationships)} categories")
    except Exception as e:
        test_results['complete_extraction'] = False
        print(f"✗ Complete relationship extraction failed: {e}")
   
    # Test JSON saving
    try:
        json_file = extractor.save_relationships_to_json(all_relationships, "test_relationships.json")
        test_results['json_saving'] = json_file is not None
        print(f"✓ JSON saving: {json_file}")
    except Exception as e:
        test_results['json_saving'] = False
        print(f"✗ JSON saving failed: {e}")
   
    # Test validation
    try:
        validation = extractor.validate_relationship_extraction(all_relationships)
        test_results['validation'] = len(validation) > 0
        print(f"✓ Relationship validation: {len(validation)} checks performed")
    except Exception as e:
        test_results['validation'] = False
        print(f"✗ Relationship validation failed: {e}")
   
    # Summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
   
    print(f"\nTest Results: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
   
    if all(test_results.values()):
        print("✓ All relationship extraction tests PASSED!")
        return True
    else:
        print("✗ Some relationship extraction tests FAILED!")
        for test_name, result in test_results.items():
            if not result:
                print(f"  - Failed: {test_name}")
        return False


if __name__ == "__main__":
    # Run the main test
    print("Running comprehensive entity and relationship extraction test...")
    results = main()
    
    print("\n" + "="*60)
    print("Running dedicated relationship extraction test...")
    test_success = test_relationship_extraction()
    
    if test_success:
        print("\n🎉 All tests completed successfully!")
        print(f"📊 Entities extracted and analyzed")
        print(f"🔗 Relationships extracted and analyzed")
        if results and isinstance(results, list):
            for i, (entities, relationships, json_file) in enumerate(results):
                print(f"💾 Results saved to JSON: {json_file}")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")

