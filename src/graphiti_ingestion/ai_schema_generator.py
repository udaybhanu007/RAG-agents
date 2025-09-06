#!/usr/bin/env python3
"""
AI-Powered Schema Generator using Azure OpenAI

This script combines statistical analysis with Azure OpenAI to generate
intelligent schemas for CSV/Excel files. It provides:
- Fast statistical analysis for structure
- AI-powered semantic understanding for content
- Domain-specific insights and recommendations
- Optimized for performance and cost

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
from openai import AzureOpenAI
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class ColumnSchema:
    """Enhanced schema definition for a single column with AI insights."""
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
    statistical_description: str
    ai_semantic_description: Optional[str]
    ai_domain_context: Optional[str]
    ai_suggested_relationships: List[str]
    potential_entity: bool
    potential_relationship: bool
    data_quality_issues: List[str]

@dataclass
class AIDatasetSchema:
    """Complete AI-enhanced schema for a dataset."""
    file_name: str
    file_path: str
    file_size_bytes: int
    total_rows: int
    total_columns: int
    columns: List[ColumnSchema]
    ai_domain_analysis: str
    ai_suggested_entities: List[Dict[str, Any]]
    ai_suggested_relationships: List[Dict[str, Any]]
    ai_data_quality_summary: str
    ai_recommended_indices: List[str]
    statistical_quality_score: float
    ai_confidence_score: float
    generation_timestamp: str
    sample_data: List[Dict[str, Any]]
    processing_time_seconds: float

class AISchemaGenerator:
    """
    AI-powered schema generator combining statistical analysis with Azure OpenAI.
    """
    
    def __init__(self, azure_openai_endpoint: str = None, azure_openai_key: str = None, 
                 azure_openai_deployment: str = None):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
        self.max_sample_values = 5
        self.max_sample_rows = 10
        
        # Azure OpenAI configuration
        self.azure_endpoint = azure_openai_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_key = azure_openai_key or os.getenv('AZURE_OPENAI_API_KEY')
        self.azure_deployment = azure_openai_deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT', 'genai-ind-gpt-4o-mini')
        
        if not all([self.azure_endpoint, self.azure_key]):
            logger.warning("⚠️ Azure OpenAI not configured - will use statistical analysis only")
            self.ai_client = None
        else:
            try:
                self.ai_client = AzureOpenAI(
                    api_key=self.azure_key,
                    api_version="2024-12-01-preview",
                    azure_endpoint=self.azure_endpoint
                )
                logger.info("✅ Azure OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Azure OpenAI: {e}")
                self.ai_client = None
    
    def detect_data_type(self, series: pd.Series) -> tuple[str, str]:
        """Detect the data type of a pandas Series."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return "unknown", "object"
        
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return "integer", "int"
            else:
                return "float", "float"
        
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime", "datetime"
        
        if pd.api.types.is_bool_dtype(series):
            return "boolean", "bool"
        
        if series.dtype == 'object':
            try:
                pd.to_numeric(clean_series)
                return "numeric_string", "str"
            except (ValueError, TypeError):
                pass
            
            try:
                pd.to_datetime(clean_series.head(10))
                return "date_string", "str"
            except (ValueError, TypeError):
                pass
        
        return "string", "str"
    
    def analyze_column_patterns(self, series: pd.Series, column_name: str) -> Dict[str, bool]:
        """Analyze column patterns to identify potential entities and relationships."""
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
        
        # Category detection
        if len(clean_series.unique()) / len(clean_series) < 0.1 and len(clean_series.unique()) < 50:
            patterns['is_category'] = True
        
        # Measurement detection
        if pd.api.types.is_numeric_dtype(series):
            if any(keyword in column_lower for keyword in ['count', 'size', 'amount', 'value', 'score']):
                patterns['is_measurement'] = True
        
        # Coordinate detection
        if any(keyword in column_lower for keyword in ['x', 'y', 'lat', 'lon', 'coord', 'bbox']):
            patterns['is_coordinate'] = True
        
        # Label detection
        if any(keyword in column_lower for keyword in ['label', 'name', 'title', 'description']):
            patterns['is_label'] = True
        
        # Reference detection
        if any(keyword in column_lower for keyword in ['ref', 'link', 'url', 'path', 'file']):
            patterns['is_reference'] = True
        
        return patterns
    
    def calculate_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistical measures for a series."""
        stats = {}
        clean_series = series.dropna()
        
        if pd.api.types.is_numeric_dtype(series):
            stats['min_value'] = float(clean_series.min()) if len(clean_series) > 0 else None
            stats['max_value'] = float(clean_series.max()) if len(clean_series) > 0 else None
            stats['mean_value'] = float(clean_series.mean()) if len(clean_series) > 0 else None
            stats['std_deviation'] = float(clean_series.std()) if len(clean_series) > 0 else None
        else:
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
    
    async def analyze_with_ai(self, df: pd.DataFrame, statistical_summary: str) -> Dict[str, Any]:
        """Use Azure OpenAI to analyze dataset and provide semantic insights."""
        if not self.ai_client:
            logger.warning("AI analysis skipped - Azure OpenAI not available")
            return self._create_fallback_ai_analysis()
        
        try:
            # Prepare data sample for AI analysis
            sample_data = df.head(10).to_string(max_cols=10, max_rows=10)
            column_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])
            
            prompt = f"""
Analyze this dataset and provide intelligent schema insights:

DATASET SAMPLE:
{sample_data}

COLUMN TYPES:
{column_info}

STATISTICAL SUMMARY:
{statistical_summary}

Please provide a JSON response with the following structure:
{{
  "domain_analysis": "Brief description of what this dataset represents (e.g., medical imaging, financial data, etc.)",
  "suggested_entities": [
    {{
      "column_name": "column name",
      "entity_type": "what kind of entity this represents",
      "description": "semantic meaning of this column",
      "domain_context": "specific domain context"
    }}
  ],
  "suggested_relationships": [
    {{
      "from_column": "source column",
      "to_column": "target column", 
      "relationship_type": "semantic relationship name",
      "description": "why these columns are related",
      "confidence": 0.9
    }}
  ],
  "data_quality_summary": "Overall assessment of data quality and potential issues",
  "recommended_indices": ["list of columns that should be indexed for performance"],
  "confidence_score": 0.85
}}

Focus on practical insights that would help with database design and querying.
"""
            
            logger.info("🤖 Requesting AI analysis from Azure OpenAI...")
            
            response = self.ai_client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst and database designer. Provide practical, actionable insights about dataset structure and relationships."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            ai_response = response.choices[0].message.content
            logger.info("✅ AI analysis completed")
            
            # Parse JSON response
            try:
                ai_insights = json.loads(ai_response)
                return ai_insights
            except json.JSONDecodeError:
                logger.warning("⚠️ AI response was not valid JSON, extracting key insights...")
                return self._extract_insights_from_text(ai_response)
                
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            return self._create_fallback_ai_analysis()
    
    def _create_fallback_ai_analysis(self) -> Dict[str, Any]:
        """Create fallback analysis when AI is not available."""
        return {
            "domain_analysis": "Dataset domain could not be determined automatically",
            "suggested_entities": [],
            "suggested_relationships": [],
            "data_quality_summary": "Statistical analysis completed, AI semantic analysis not available",
            "recommended_indices": [],
            "confidence_score": 0.5
        }
    
    def _extract_insights_from_text(self, text: str) -> Dict[str, Any]:
        """Extract insights from non-JSON AI response."""
        return {
            "domain_analysis": "AI provided text analysis (see full response in logs)",
            "suggested_entities": [],
            "suggested_relationships": [],
            "data_quality_summary": text[:200] + "..." if len(text) > 200 else text,
            "recommended_indices": [],
            "confidence_score": 0.6
        }
    
    def generate_statistical_summary(self, df: pd.DataFrame) -> str:
        """Generate a statistical summary for AI analysis."""
        summary = []
        summary.append(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
        summary.append(f"Missing values: {df.isnull().sum().sum()} total")
        
        for col in df.columns:
            unique_count = df[col].nunique()
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            summary.append(f"{col}: {unique_count} unique values, {missing_pct:.1f}% missing")
        
        return "\n".join(summary)
    
    async def analyze_file(self, file_path: str) -> AIDatasetSchema:
        """Analyze a CSV/Excel file and generate AI-enhanced schema."""
        start_time = time.time()
        logger.info(f"🔍 Analyzing file: {file_path}")
        
        # Read the file
        file_extension = Path(file_path).suffix.lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"📊 Loaded dataset: {len(df)} rows × {len(df.columns)} columns")
        
        # Generate statistical summary
        statistical_summary = self.generate_statistical_summary(df)
        
        # Get AI insights
        ai_insights = await self.analyze_with_ai(df, statistical_summary)
        
        # Analyze each column with AI enhancement
        columns_info = []
        for column_name in df.columns:
            series = df[column_name]
            
            # Statistical analysis
            data_type, python_type = self.detect_data_type(series)
            missing_count = series.isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            unique_values = series.nunique()
            
            clean_series = series.dropna()
            sample_values = clean_series.head(self.max_sample_values).tolist() if len(clean_series) > 0 else []
            
            stats = self.calculate_statistics(series)
            patterns = self.analyze_column_patterns(series, column_name)
            
            # Generate statistical description
            statistical_desc = self.generate_statistical_description(column_name, data_type, patterns, unique_values, len(df))
            
            # Find AI insights for this column
            ai_semantic_desc = None
            ai_domain_context = None
            ai_suggested_rels = []
            
            for entity in ai_insights.get('suggested_entities', []):
                if entity.get('column_name') == column_name:
                    ai_semantic_desc = entity.get('description')
                    ai_domain_context = entity.get('domain_context')
                    break
            
            # Find related relationships
            for rel in ai_insights.get('suggested_relationships', []):
                if rel.get('from_column') == column_name or rel.get('to_column') == column_name:
                    ai_suggested_rels.append(f"{rel.get('relationship_type')}: {rel.get('description')}")
            
            # Determine entity/relationship potential
            potential_entity = patterns['is_id_column'] or (unique_values / len(df) > 0.8 and unique_values > 10)
            potential_relationship = patterns['is_reference'] or patterns['is_category']
            
            # Identify data quality issues
            quality_issues = []
            if missing_percentage > 10:
                quality_issues.append(f"High missing rate: {missing_percentage:.1f}%")
            if unique_values == 1:
                quality_issues.append("No variance - all values are identical")
            if data_type == "string" and unique_values == len(df):
                quality_issues.append("Potential over-normalization - all values unique")
            
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
                statistical_description=statistical_desc,
                ai_semantic_description=ai_semantic_desc,
                ai_domain_context=ai_domain_context,
                ai_suggested_relationships=ai_suggested_rels,
                potential_entity=potential_entity,
                potential_relationship=potential_relationship,
                data_quality_issues=quality_issues
            )
            
            columns_info.append(column_schema)
        
        # Calculate statistical quality score
        statistical_quality = self.calculate_statistical_quality_score(df, columns_info)
        
        # Generate sample data
        sample_data = [{str(k): v for k, v in row.items()} for row in df.head(self.max_sample_rows).to_dict('records')]
        
        # Create enhanced schema
        processing_time = time.time() - start_time
        
        schema = AIDatasetSchema(
            file_name=Path(file_path).name,
            file_path=file_path,
            file_size_bytes=os.path.getsize(file_path),
            total_rows=len(df),
            total_columns=len(df.columns),
            columns=columns_info,
            ai_domain_analysis=ai_insights.get('domain_analysis', 'Not analyzed'),
            ai_suggested_entities=ai_insights.get('suggested_entities', []),
            ai_suggested_relationships=ai_insights.get('suggested_relationships', []),
            ai_data_quality_summary=ai_insights.get('data_quality_summary', 'Not analyzed'),
            ai_recommended_indices=ai_insights.get('recommended_indices', []),
            statistical_quality_score=round(statistical_quality, 3),
            ai_confidence_score=ai_insights.get('confidence_score', 0.5),
            generation_timestamp=datetime.now().isoformat(),
            sample_data=sample_data,
            processing_time_seconds=round(processing_time, 2)
        )
        
        logger.info(f"✅ AI-enhanced schema analysis completed")
        logger.info(f"   📊 Statistical quality: {statistical_quality:.3f}")
        logger.info(f"   🤖 AI confidence: {ai_insights.get('confidence_score', 0.5):.3f}")
        logger.info(f"   ⏱️ Processing time: {processing_time:.2f}s")
        
        return schema
    
    def generate_statistical_description(self, column_name: str, data_type: str, patterns: Dict[str, bool], unique_values: int, total_rows: int) -> str:
        """Generate a statistical description for a column."""
        description_parts = []
        
        description_parts.append(f"{data_type.replace('_', ' ').title()} column")
        
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
        
        cardinality_ratio = unique_values / total_rows if total_rows > 0 else 0
        if cardinality_ratio > 0.9:
            description_parts.append("with high cardinality (mostly unique values)")
        elif cardinality_ratio < 0.1:
            description_parts.append("with low cardinality (many repeated values)")
        else:
            description_parts.append("with moderate cardinality")
        
        return " ".join(description_parts) + "."
    
    def calculate_statistical_quality_score(self, df: pd.DataFrame, columns_info: List[ColumnSchema]) -> float:
        """Calculate statistical data quality score."""
        scores = []
        
        for col_info in columns_info:
            completeness = 1 - (col_info.missing_percentage / 100)
            scores.append(completeness)
        
        total_rows = len(df)
        for col_info in columns_info:
            if total_rows > 0:
                uniqueness = col_info.unique_values / total_rows
                scores.append(min(uniqueness * 2, 1.0))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def save_schema_to_json(self, schema: AIDatasetSchema, output_path: str) -> None:
        """Save the AI-enhanced schema to a JSON file."""
        schema_dict = asdict(schema)
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ AI-enhanced schema saved to: {output_path}")
    
    def generate_schema_report(self, schema: AIDatasetSchema) -> str:
        """Generate a comprehensive AI-enhanced report."""
        report = []
        report.append("="*80)
        report.append(f"AI-ENHANCED DATASET SCHEMA REPORT: {schema.file_name}")
        report.append("="*80)
        report.append(f"File Path: {schema.file_path}")
        report.append(f"File Size: {schema.file_size_bytes:,} bytes")
        report.append(f"Generated: {schema.generation_timestamp}")
        report.append(f"Processing Time: {schema.processing_time_seconds}s")
        report.append(f"Statistical Quality Score: {schema.statistical_quality_score}/1.0")
        report.append(f"AI Confidence Score: {schema.ai_confidence_score}/1.0")
        report.append("")
        
        report.append("DATASET OVERVIEW:")
        report.append(f"  Rows: {schema.total_rows:,}")
        report.append(f"  Columns: {schema.total_columns}")
        report.append("")
        
        report.append("AI DOMAIN ANALYSIS:")
        report.append(f"  {schema.ai_domain_analysis}")
        report.append("")
        
        report.append("AI DATA QUALITY ASSESSMENT:")
        report.append(f"  {schema.ai_data_quality_summary}")
        report.append("")
        
        if schema.ai_recommended_indices:
            report.append("AI RECOMMENDED INDICES:")
            for index in schema.ai_recommended_indices:
                report.append(f"  - {index}")
            report.append("")
        
        report.append("COLUMN DETAILS:")
        for col in schema.columns:
            report.append(f"  {col.name}:")
            report.append(f"    Type: {col.data_type} ({col.python_type})")
            report.append(f"    Unique Values: {col.unique_values:,}")
            report.append(f"    Missing: {col.missing_count} ({col.missing_percentage}%)")
            report.append(f"    Statistical: {col.statistical_description}")
            if col.ai_semantic_description:
                report.append(f"    AI Semantic: {col.ai_semantic_description}")
            if col.ai_domain_context:
                report.append(f"    Domain Context: {col.ai_domain_context}")
            if col.data_quality_issues:
                report.append(f"    Quality Issues: {', '.join(col.data_quality_issues)}")
            if col.sample_values:
                report.append(f"    Sample Values: {col.sample_values}")
            report.append("")
        
        if schema.ai_suggested_entities:
            report.append("AI SUGGESTED ENTITIES:")
            for entity in schema.ai_suggested_entities:
                report.append(f"  - {entity.get('column_name')}: {entity.get('entity_type')} ({entity.get('description')})")
            report.append("")
        
        if schema.ai_suggested_relationships:
            report.append("AI SUGGESTED RELATIONSHIPS:")
            for rel in schema.ai_suggested_relationships:
                conf = rel.get('confidence', 0)
                report.append(f"  - {rel.get('from_column')} -> {rel.get('relationship_type')} -> {rel.get('to_column')} (confidence: {conf})")
                report.append(f"    Description: {rel.get('description')}")
            report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)

async def main():
    """Main execution function with AI-enhanced schema generation."""
    parser = argparse.ArgumentParser(description="Generate AI-enhanced schema for CSV/Excel files")
    parser.add_argument("input_file", help="Path to the CSV/Excel file to analyze")
    parser.add_argument("-o", "--output", help="Output JSON file path (default: auto-generated)")
    parser.add_argument("-r", "--report", help="Generate text report file", action="store_true")
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")
    parser.add_argument("--azure-endpoint", help="Azure OpenAI endpoint URL")
    parser.add_argument("--azure-key", help="Azure OpenAI API key")
    parser.add_argument("--azure-deployment", help="Azure OpenAI deployment name")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize AI schema generator
        generator = AISchemaGenerator(
            azure_openai_endpoint=args.azure_endpoint,
            azure_openai_key=args.azure_key,
            azure_openai_deployment=args.azure_deployment
        )
        
        # Validate input file
        if not os.path.exists(args.input_file):
            logger.error(f"❌ Input file not found: {args.input_file}")
            return 1
        
        # Generate output path if not provided
        if not args.output:
            input_path = Path(args.input_file)
            args.output = str(input_path.parent / f"{input_path.stem}_ai_schema.json")
        
        # Analyze the file with AI
        logger.info("🚀 Starting AI-enhanced schema analysis...")
        schema = await generator.analyze_file(args.input_file)
        
        # Save JSON schema
        generator.save_schema_to_json(schema, args.output)
        
        # Generate and save report if requested
        if args.report:
            report_path = Path(args.output).with_suffix('.txt')
            report_content = generator.generate_schema_report(schema)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"✅ AI-enhanced report saved to: {report_path}")
            
            # Print summary to console
            print("\n" + "="*60)
            print("AI-ENHANCED SCHEMA ANALYSIS SUMMARY")
            print("="*60)
            print(f"File: {schema.file_name}")
            print(f"Rows: {schema.total_rows:,}")
            print(f"Columns: {schema.total_columns}")
            print(f"Statistical Quality: {schema.statistical_quality_score}/1.0")
            print(f"AI Confidence: {schema.ai_confidence_score}/1.0")
            print(f"Domain: {schema.ai_domain_analysis}")
            print(f"Processing Time: {schema.processing_time_seconds}s")
            print(f"Schema saved to: {args.output}")
            print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error during AI-enhanced schema generation: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
