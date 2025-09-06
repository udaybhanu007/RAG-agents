#!/usr/bin/env python3
"""
Dynamic Schema Generator for CSV/Excel files using Graphiti Core

This script analyzes CSV/Excel files and generates a comprehensive schema
that can be saved in JSON format. The script is designed to be reusable
across different types of documents.

Features:
- Automatic column type detection
- Statistical analysis of data
- Entity and relationship inference
- JSON schema export
- Support for CSV and Excel files
- Dynamic configuration for different file types

Author: AI Assistant
Date: September 2, 2025
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass, asdict
import argparse

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ColumnSchema:
    """Schema definition for a single column."""
    name: str
    data_type: str
    python_type: str
    nullable: bool
    unique_values: int
    sample_values: List[Any]
    min_value: Optional[Union[int, float, str]]
    max_value: Optional[Union[int, float, str]]
    mean_value: Optional[float]
    std_deviation: Optional[float]
    missing_count: int
    missing_percentage: float
    description: str
    potential_entity: bool
    potential_relationship: bool

@dataclass
class DatasetSchema:
    """Complete schema for a dataset."""
    file_name: str
    file_path: str
    file_size_bytes: int
    total_rows: int
    total_columns: int
    columns: List[ColumnSchema]
    potential_entities: List[str]
    potential_relationships: List[Dict[str, str]]
    data_quality_score: float
    generation_timestamp: str
    sample_data: List[Dict[str, Any]]

class DynamicSchemaGenerator:
    """
    Dynamic schema generator for CSV/Excel files using statistical analysis
    and pattern recognition to infer entity-relationship structures.
    """
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
        self.max_sample_values = 5
        self.max_sample_rows = 10
        
    def detect_data_type(self, series: pd.Series) -> tuple[str, str]:
        """
        Detect the data type of a pandas Series.
        
        Returns:
            tuple: (semantic_type, python_type)
        """
        # Remove null values for type detection
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return "unknown", "object"
        
        # Check for numeric types
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return "integer", "int"
            else:
                return "float", "float"
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime", "datetime"
        
        # Check for boolean
        if pd.api.types.is_bool_dtype(series):
            return "boolean", "bool"
        
        # Check if string column contains numbers
        if series.dtype == 'object':
            try:
                pd.to_numeric(clean_series)
                return "numeric_string", "str"
            except (ValueError, TypeError):
                pass
            
            # Check for date strings
            try:
                pd.to_datetime(clean_series.head(10))
                return "date_string", "str"
            except (ValueError, TypeError):
                pass
        
        # Default to string
        return "string", "str"
    
    def analyze_column_patterns(self, series: pd.Series, column_name: str) -> Dict[str, bool]:
        """
        Analyze column patterns to identify potential entities and relationships.
        
        Args:
            series: Pandas series to analyze
            column_name: Name of the column
            
        Returns:
            Dict with pattern analysis results
        """
        patterns = {
            'is_id_column': False,
            'is_category': False,
            'is_measurement': False,
            'is_coordinate': False,
            'is_label': False,
            'is_reference': False
        }
        
        column_lower = column_name.lower()
        clean_series = series.dropna()
        
        # ID column detection
        if any(keyword in column_lower for keyword in ['id', 'index', 'key']):
            patterns['is_id_column'] = True
        
        # Category detection (low cardinality, string type)
        if len(clean_series.unique()) / len(clean_series) < 0.1 and len(clean_series.unique()) < 50:
            patterns['is_category'] = True
        
        # Measurement detection (numeric with descriptive name)
        if pd.api.types.is_numeric_dtype(series):
            if any(keyword in column_lower for keyword in ['count', 'size', 'amount', 'value', 'score']):
                patterns['is_measurement'] = True
        
        # Coordinate detection
        if any(keyword in column_lower for keyword in ['x', 'y', 'lat', 'lon', 'coord', 'bbox']):
            patterns['is_coordinate'] = True
        
        # Label detection
        if any(keyword in column_lower for keyword in ['label', 'name', 'title', 'description']):
            patterns['is_label'] = True
        
        # Reference detection (points to other entities)
        if any(keyword in column_lower for keyword in ['ref', 'link', 'url', 'path', 'file']):
            patterns['is_reference'] = True
        
        return patterns
    
    def calculate_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """
        Calculate statistical measures for a series.
        
        Args:
            series: Pandas series to analyze
            
        Returns:
            Dictionary of statistical measures
        """
        stats = {}
        clean_series = series.dropna()
        
        if pd.api.types.is_numeric_dtype(series):
            stats['min_value'] = float(clean_series.min()) if len(clean_series) > 0 else None
            stats['max_value'] = float(clean_series.max()) if len(clean_series) > 0 else None
            stats['mean_value'] = float(clean_series.mean()) if len(clean_series) > 0 else None
            stats['std_deviation'] = float(clean_series.std()) if len(clean_series) > 0 else None
        else:
            # For non-numeric data, use string length as proxy
            if len(clean_series) > 0:
                str_lengths = clean_series.astype(str).str.len()
                min_idx = str_lengths.idxmin()
                max_idx = str_lengths.idxmax()
                stats['min_value'] = str(clean_series.loc[min_idx]) if min_idx is not None else None
                stats['max_value'] = str(clean_series.loc[max_idx]) if max_idx is not None else None
                stats['mean_value'] = float(str_lengths.mean())
                stats['std_deviation'] = float(str_lengths.std())
            else:
                stats['min_value'] = None
                stats['max_value'] = None
                stats['mean_value'] = None
                stats['std_deviation'] = None
        
        return stats
    
    def infer_entities_and_relationships(self, df: pd.DataFrame, columns_info: List[ColumnSchema]) -> tuple[List[str], List[Dict[str, str]]]:
        """
        Infer potential entities and relationships from the dataset.
        
        Args:
            df: The dataframe to analyze
            columns_info: List of column schemas
            
        Returns:
            tuple: (potential_entities, potential_relationships)
        """
        entities = []
        relationships = []
        
        # Find potential entity columns (ID columns, unique identifiers)
        for col_info in columns_info:
            if col_info.potential_entity:
                entities.append(col_info.name)
        
        # Infer relationships based on column patterns
        id_columns = [col.name for col in columns_info if 'id' in col.name.lower() or 'index' in col.name.lower()]
        reference_columns = [col.name for col in columns_info if col.potential_relationship]
        
        # Create relationships between entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                relationships.append({
                    'from_entity': entity1,
                    'to_entity': entity2,
                    'relationship_type': 'RELATED_TO',
                    'confidence': 0.5
                })
        
        # Add specific relationships for known patterns
        if 'Image Index' in df.columns and 'Finding Label' in df.columns:
            relationships.append({
                'from_entity': 'Image Index',
                'to_entity': 'Finding Label',
                'relationship_type': 'HAS_FINDING',
                'confidence': 0.9
            })
        
        return entities, relationships
    
    def calculate_data_quality_score(self, df: pd.DataFrame, columns_info: List[ColumnSchema]) -> float:
        """
        Calculate an overall data quality score.
        
        Args:
            df: The dataframe to analyze
            columns_info: List of column schemas
            
        Returns:
            Quality score between 0 and 1
        """
        scores = []
        
        for col_info in columns_info:
            # Completeness score (1 - missing_percentage)
            completeness = 1 - (col_info.missing_percentage / 100)
            scores.append(completeness)
        
        # Uniqueness score (penalize columns with too many duplicates)
        total_rows = len(df)
        for col_info in columns_info:
            if total_rows > 0:
                uniqueness = col_info.unique_values / total_rows
                scores.append(min(uniqueness * 2, 1.0))  # Cap at 1.0
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def analyze_file(self, file_path: str) -> DatasetSchema:
        """
        Analyze a CSV/Excel file and generate a comprehensive schema.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            DatasetSchema object with complete analysis
        """
        logger.info(f"Analyzing file: {file_path}")
        
        # Read the file
        file_extension = Path(file_path).suffix.lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Analyze each column
        columns_info = []
        for column_name in df.columns:
            series = df[column_name]
            
            # Basic information
            data_type, python_type = self.detect_data_type(series)
            missing_count = series.isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            unique_values = series.nunique()
            
            # Sample values (excluding nulls)
            clean_series = series.dropna()
            sample_values = clean_series.head(self.max_sample_values).tolist() if len(clean_series) > 0 else []
            
            # Statistical analysis
            stats = self.calculate_statistics(series)
            
            # Pattern analysis
            patterns = self.analyze_column_patterns(series, column_name)
            
            # Generate description
            description = self.generate_column_description(column_name, data_type, patterns, unique_values, len(df))
            
            # Determine if this could be an entity or relationship
            potential_entity = patterns['is_id_column'] or (unique_values / len(df) > 0.8 and unique_values > 10)
            potential_relationship = patterns['is_reference'] or patterns['is_category']
            
            column_schema = ColumnSchema(
                name=column_name,
                data_type=data_type,
                python_type=python_type,
                nullable=missing_count > 0,
                unique_values=unique_values,
                sample_values=sample_values,
                min_value=stats['min_value'],
                max_value=stats['max_value'],
                mean_value=stats['mean_value'],
                std_deviation=stats['std_deviation'],
                missing_count=int(missing_count),
                missing_percentage=round(missing_percentage, 2),
                description=description,
                potential_entity=potential_entity,
                potential_relationship=potential_relationship
            )
            
            columns_info.append(column_schema)
        
        # Infer entities and relationships
        entities, relationships = self.infer_entities_and_relationships(df, columns_info)
        
        # Calculate data quality score
        quality_score = self.calculate_data_quality_score(df, columns_info)
        
        # Generate sample data
        sample_data = [{str(k): v for k, v in row.items()} for row in df.head(self.max_sample_rows).to_dict('records')]
        
        # Create final schema
        schema = DatasetSchema(
            file_name=Path(file_path).name,
            file_path=file_path,
            file_size_bytes=os.path.getsize(file_path),
            total_rows=len(df),
            total_columns=len(df.columns),
            columns=columns_info,
            potential_entities=entities,
            potential_relationships=relationships,
            data_quality_score=round(quality_score, 3),
            generation_timestamp=datetime.now().isoformat(),
            sample_data=sample_data
        )
        
        logger.info(f"✅ Schema analysis completed for {schema.file_name}")
        logger.info(f"   Entities identified: {len(entities)}")
        logger.info(f"   Relationships identified: {len(relationships)}")
        logger.info(f"   Data quality score: {quality_score:.3f}")
        
        return schema
    
    def generate_column_description(self, column_name: str, data_type: str, patterns: Dict[str, bool], unique_values: int, total_rows: int) -> str:
        """
        Generate a human-readable description for a column.
        
        Args:
            column_name: Name of the column
            data_type: Detected data type
            patterns: Pattern analysis results
            unique_values: Number of unique values
            total_rows: Total number of rows
            
        Returns:
            Description string
        """
        description_parts = []
        
        # Basic type description
        description_parts.append(f"{data_type.replace('_', ' ').title()} column")
        
        # Add pattern-based descriptions
        if patterns['is_id_column']:
            description_parts.append("serving as an identifier")
        elif patterns['is_category']:
            description_parts.append("containing categorical data")
        elif patterns['is_measurement']:
            description_parts.append("containing measurement values")
        elif patterns['is_coordinate']:
            description_parts.append("containing coordinate or positional data")
        elif patterns['is_label']:
            description_parts.append("containing label or descriptive text")
        elif patterns['is_reference']:
            description_parts.append("containing references to external resources")
        
        # Add cardinality information
        cardinality_ratio = unique_values / total_rows if total_rows > 0 else 0
        if cardinality_ratio > 0.9:
            description_parts.append("with high cardinality (mostly unique values)")
        elif cardinality_ratio < 0.1:
            description_parts.append("with low cardinality (many repeated values)")
        else:
            description_parts.append("with moderate cardinality")
        
        return " ".join(description_parts) + "."
    
    def save_schema_to_json(self, schema: DatasetSchema, output_path: str) -> None:
        """
        Save the schema to a JSON file.
        
        Args:
            schema: DatasetSchema object to save
            output_path: Path where to save the JSON file
        """
        # Convert dataclass to dictionary
        schema_dict = asdict(schema)
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON with pretty formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Schema saved to: {output_path}")
    
    def generate_schema_report(self, schema: DatasetSchema) -> str:
        """
        Generate a human-readable report of the schema.
        
        Args:
            schema: DatasetSchema object
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*80)
        report.append(f"DATASET SCHEMA REPORT: {schema.file_name}")
        report.append("="*80)
        report.append(f"File Path: {schema.file_path}")
        report.append(f"File Size: {schema.file_size_bytes:,} bytes")
        report.append(f"Generated: {schema.generation_timestamp}")
        report.append(f"Data Quality Score: {schema.data_quality_score}/1.0")
        report.append("")
        
        report.append(f"DATASET OVERVIEW:")
        report.append(f"  Rows: {schema.total_rows:,}")
        report.append(f"  Columns: {schema.total_columns}")
        report.append("")
        
        report.append("COLUMN DETAILS:")
        for col in schema.columns:
            report.append(f"  {col.name}:")
            report.append(f"    Type: {col.data_type} ({col.python_type})")
            report.append(f"    Unique Values: {col.unique_values:,}")
            report.append(f"    Missing: {col.missing_count} ({col.missing_percentage}%)")
            report.append(f"    Description: {col.description}")
            if col.sample_values:
                report.append(f"    Sample Values: {col.sample_values}")
            report.append("")
        
        if schema.potential_entities:
            report.append("POTENTIAL ENTITIES:")
            for entity in schema.potential_entities:
                report.append(f"  - {entity}")
            report.append("")
        
        if schema.potential_relationships:
            report.append("POTENTIAL RELATIONSHIPS:")
            for rel in schema.potential_relationships:
                report.append(f"  - {rel['from_entity']} -> {rel['relationship_type']} -> {rel['to_entity']} (confidence: {rel['confidence']})")
            report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)

def process_multiple_files(file_paths: List[str], output_dir: str = None, generate_reports: bool = False) -> Dict[str, Any]:
    """
    Process multiple Excel/CSV files and generate schemas for each.
    
    Args:
        file_paths: List of file paths to process
        output_dir: Directory to save schemas (default: same as input files)
        generate_reports: Whether to generate text reports
        
    Returns:
        Dictionary with processing results
    """
    generator = DynamicSchemaGenerator()
    results = {
        'processed_files': [],
        'failed_files': [],
        'total_files': len(file_paths),
        'schemas': [],
        'summary': {}
    }
    
    logger.info(f"🔄 Processing {len(file_paths)} files...")
    
    for i, file_path in enumerate(file_paths, 1):
        try:
            logger.info(f"[{i}/{len(file_paths)}] Processing: {Path(file_path).name}")
            
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ File not found: {file_path}")
                results['failed_files'].append({'file': file_path, 'error': 'File not found'})
                continue
            
            # Generate schema
            schema = generator.analyze_file(file_path)
            
            # Determine output path
            input_path = Path(file_path)
            if output_dir:
                output_path = Path(output_dir) / f"{input_path.stem}_schema.json"
                report_path = Path(output_dir) / f"{input_path.stem}_report.txt"
            else:
                output_path = input_path.parent / f"{input_path.stem}_schema.json"
                report_path = input_path.parent / f"{input_path.stem}_report.txt"
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save schema
            generator.save_schema_to_json(schema, str(output_path))
            
            # Generate report if requested
            if generate_reports:
                report_content = generator.generate_schema_report(schema)
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"📄 Report saved: {report_path.name}")
            
            # Track results
            results['processed_files'].append({
                'file': file_path,
                'schema_path': str(output_path),
                'report_path': str(report_path) if generate_reports else None,
                'rows': schema.total_rows,
                'columns': schema.total_columns,
                'quality_score': schema.data_quality_score,
                'entities': len(schema.potential_entities),
                'relationships': len(schema.potential_relationships)
            })
            results['schemas'].append(schema)
            
            logger.info(f"✅ Completed: {input_path.name} (Score: {schema.data_quality_score:.3f})")
            
        except Exception as e:
            logger.error(f"❌ Failed processing {file_path}: {e}")
            results['failed_files'].append({'file': file_path, 'error': str(e)})
    
    # Generate summary
    if results['processed_files']:
        total_rows = sum(f['rows'] for f in results['processed_files'])
        total_columns = sum(f['columns'] for f in results['processed_files'])
        avg_quality = sum(f['quality_score'] for f in results['processed_files']) / len(results['processed_files'])
        total_entities = sum(f['entities'] for f in results['processed_files'])
        total_relationships = sum(f['relationships'] for f in results['processed_files'])
        
        results['summary'] = {
            'successful_files': len(results['processed_files']),
            'failed_files': len(results['failed_files']),
            'total_rows_processed': total_rows,
            'total_columns_processed': total_columns,
            'average_quality_score': avg_quality,
            'total_entities_found': total_entities,
            'total_relationships_found': total_relationships
        }
    
    return results

def find_excel_files(directory: str, recursive: bool = True) -> List[str]:
    """Find all Excel/CSV files in a directory."""
    directory_path = Path(directory)
    files = []
    
    patterns = ['*.csv', '*.xlsx', '*.xls']
    
    for pattern in patterns:
        if recursive:
            files.extend(directory_path.rglob(pattern))
        else:
            files.extend(directory_path.glob(pattern))
    
    return [str(f) for f in files]

def main():
    """Main execution function with command line interface."""
    parser = argparse.ArgumentParser(description="Generate schema for CSV/Excel files using Graphiti core")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single CSV/Excel file to analyze")
    group.add_argument("--directory", help="Directory containing Excel/CSV files to process")
    group.add_argument("--files", nargs='+', help="List of specific files to process")
    
    # Output options
    parser.add_argument("-o", "--output", help="Output directory or file path (default: auto-generated)")
    parser.add_argument("-r", "--report", help="Generate text report files", action="store_true")
    parser.add_argument("--recursive", help="Search directories recursively", action="store_true", default=True)
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize schema generator
        generator = DynamicSchemaGenerator()
        
        # Determine input files
        if args.file:
            # Single file mode
            file_paths = [args.file]
        elif args.directory:
            # Directory mode
            if not os.path.exists(args.directory):
                logger.error(f"❌ Directory not found: {args.directory}")
                return 1
            file_paths = find_excel_files(args.directory, args.recursive)
            if not file_paths:
                logger.error(f"❌ No Excel/CSV files found in: {args.directory}")
                return 1
            logger.info(f"📁 Found {len(file_paths)} files in directory")
        else:
            # Multiple files mode
            file_paths = args.files
        
        # Process single file (backward compatibility)
        if len(file_paths) == 1 and args.file:
            file_path = file_paths[0]
            
            # Validate input file
            if not os.path.exists(file_path):
                logger.error(f"❌ Input file not found: {file_path}")
                return 1
            
            # Generate output path if not provided
            if not args.output:
                input_path = Path(file_path)
                args.output = str(input_path.parent / f"{input_path.stem}_schema.json")
            
            # Analyze the file
            logger.info("Starting schema analysis...")
            schema = generator.analyze_file(file_path)
            
            # Save JSON schema
            generator.save_schema_to_json(schema, args.output)
        
            
            # Generate and save report if requested
            if args.report:
                report_path = Path(args.output).with_suffix('.txt')
                report_content = generator.generate_schema_report(schema)
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"✅ Report saved to: {report_path}")
                
                # Also print summary to console
                print("\n" + "="*60)
                print("SCHEMA ANALYSIS SUMMARY")
                print("="*60)
                print(f"File: {schema.file_name}")
                print(f"Rows: {schema.total_rows:,}")
                print(f"Columns: {schema.total_columns}")
                print(f"Quality Score: {schema.data_quality_score}/1.0")
                print(f"Entities: {len(schema.potential_entities)}")
                print(f"Relationships: {len(schema.potential_relationships)}")
                print(f"Schema saved to: {args.output}")
                print("="*60)
        
        # Process multiple files
        else:
            # Determine output directory
            output_dir = args.output if args.output else None
            
            # Process all files
            results = process_multiple_files(file_paths, output_dir, args.report)
            
            # Print summary
            print("\n" + "="*80)
            print("BATCH PROCESSING SUMMARY")
            print("="*80)
            print(f"Total files processed: {results['summary'].get('successful_files', 0)}")
            print(f"Failed files: {results['summary'].get('failed_files', 0)}")
            print(f"Total rows processed: {results['summary'].get('total_rows_processed', 0):,}")
            print(f"Total columns processed: {results['summary'].get('total_columns_processed', 0)}")
            print(f"Average quality score: {results['summary'].get('average_quality_score', 0):.3f}/1.0")
            print(f"Total entities found: {results['summary'].get('total_entities_found', 0)}")
            print(f"Total relationships found: {results['summary'].get('total_relationships_found', 0)}")
            print("="*80)
            
            # Show processed files
            if results['processed_files']:
                print("\n📊 PROCESSED FILES:")
                for file_info in results['processed_files']:
                    file_name = Path(file_info['file']).name
                    print(f"  ✅ {file_name}")
                    print(f"     Rows: {file_info['rows']:,} | Columns: {file_info['columns']} | Score: {file_info['quality_score']:.3f}")
                    print(f"     Entities: {file_info['entities']} | Relationships: {file_info['relationships']}")
                    print(f"     Schema: {Path(file_info['schema_path']).name}")
                    if file_info['report_path']:
                        print(f"     Report: {Path(file_info['report_path']).name}")
                    print()
            
            # Show failed files
            if results['failed_files']:
                print("\n❌ FAILED FILES:")
                for failed in results['failed_files']:
                    print(f"  - {Path(failed['file']).name}: {failed['error']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error during schema generation: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())
