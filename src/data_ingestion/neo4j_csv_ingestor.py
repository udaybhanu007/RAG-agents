"""
Neo4j ingestion module for CSV-extracted schema and data
"""
import pandas as pd
import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from neo4j import GraphDatabase, Driver, Session
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    Driver = None
    Session = None

# Handle imports for different execution contexts
try:
    # Try relative imports first (when run as module)
    from ..core.azure_keyvault_manager import get_secret_from_keyvault
    from .ingestion_structured_document import StructuredDocumentIngestor, CSVSchemaExtraction
except ImportError:
    try:
        # Try absolute imports (when run directly)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from core.azure_keyvault_manager import get_secret_from_keyvault
        from data_ingestion.ingestion_structured_document import StructuredDocumentIngestor, CSVSchemaExtraction
    except ImportError:
        # Final fallback - set up path manually
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_file_dir)
        sys.path.insert(0, parent_dir)
        
        from core.azure_keyvault_manager import get_secret_from_keyvault
        from data_ingestion.ingestion_structured_document import StructuredDocumentIngestor, CSVSchemaExtraction

logger = logging.getLogger(__name__)

@dataclass
class Neo4jIngestionResult:
    """Result of Neo4j ingestion operation"""
    success: bool
    nodes_created: int
    relationships_created: int
    constraints_created: int
    indexes_created: int
    execution_time: float
    errors: List[str]

class Neo4jCSVIngestor:
    """
    Specialized class for ingesting CSV-extracted schemas and data into Neo4j
    """
    
    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        """Initialize Neo4j connection"""
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        # Get credentials from keyvault (which falls back to environment variables when disabled)
        self.uri = uri or get_secret_from_keyvault("NEO4J_URI") or "neo4j://localhost:7687"
        self.username = username or get_secret_from_keyvault("NEO4J_USERNAME") or "neo4j"
        self.password = password or get_secret_from_keyvault("NEO4J_PASSWORD") or "password"
        
        logger.info(f"Initializing Neo4j connection to: {self.uri}")
        logger.info(f"Using username: {self.username}")
        
        self.driver: Optional[Any] = None
        self.schema_ingestor = StructuredDocumentIngestor()
        self._connect()
    
    def _connect(self):
        """Establish Neo4j connection"""
        try:
            if GraphDatabase is not None:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
                # Test connection
                with self.driver.session() as session:
                    session.run("RETURN 1")
                logger.info(f"Successfully connected to Neo4j at {self.uri}")
            else:
                raise ImportError("Neo4j driver not available")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def ingest_csv_to_neo4j(self, csv_file_path: str, clear_existing: bool = False) -> Neo4jIngestionResult:
        """
        Complete pipeline: Extract schema from CSV and ingest to Neo4j
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting CSV to Neo4j ingestion: {csv_file_path}")
        
        result = Neo4jIngestionResult(
            success=False,
            nodes_created=0,
            relationships_created=0,
            constraints_created=0,
            indexes_created=0,
            execution_time=0.0,
            errors=[]
        )
        
        try:
            # Step 1: Extract schema using LLM
            logger.info("Extracting schema using LLM...")
            schema = self.schema_ingestor.get_schema_for_csv(csv_file_path)
            if not schema:
                result.errors.append("Failed to extract schema from CSV")
                return result
            
            logger.info(f"Schema extracted: {len(schema.nodes)} nodes, {len(schema.relationships)} relationships")
            
            # Step 2: Load CSV data
            df = pd.read_csv(csv_file_path)
            logger.info(f"CSV loaded: {len(df)} rows")
            
            # Step 3: Clear existing data if requested
            if clear_existing:
                self._clear_database()
            
            # Step 4: Create constraints and indexes
            constraints_created = self._create_constraints(schema)
            indexes_created = self._create_indexes(schema)
            result.constraints_created = constraints_created
            result.indexes_created = indexes_created
            
            # Step 5: Ingest nodes
            nodes_created = self._ingest_nodes(schema, df)
            result.nodes_created = nodes_created
            
            # Step 6: Ingest relationships
            relationships_created = self._ingest_relationships(schema, df)
            result.relationships_created = relationships_created
            
            result.success = True
            logger.info(f"Ingestion completed successfully: {nodes_created} nodes, {relationships_created} relationships")
            
        except Exception as e:
            error_msg = f"Ingestion failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    def _clear_database(self):
        """Clear all nodes and relationships from the database"""
        if not self.driver:
            logger.error("Driver not initialized")
            return
            
        logger.warning("Clearing entire Neo4j database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")
    
    def _create_constraints(self, schema: CSVSchemaExtraction) -> int:
        """Create unique constraints for node types"""
        constraints_created = 0
        
        if not self.driver:
            logger.error("Driver not initialized")
            return 0
        
        with self.driver.session() as session:
            for node in schema.nodes:
                if node.unique_key:
                    # Clean up property name for Neo4j
                    clean_property = self._clean_property_name(node.unique_key)
                    constraint_name = f"unique_{node.node_type.lower()}_{clean_property}"
                    
                    cypher = f"""
                    CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                    FOR (n:{node.node_type})
                    REQUIRE n.{clean_property} IS UNIQUE
                    """
                    
                    try:
                        # Execute constraint creation
                        session.run(cypher)
                        constraints_created += 1
                        logger.info(f"Created constraint: {constraint_name}")
                    except Exception as e:
                        logger.warning(f"Failed to create constraint {constraint_name}: {e}")
        
        return constraints_created
    
    def _create_indexes(self, schema: CSVSchemaExtraction) -> int:
        """Create indexes for better query performance"""
        indexes_created = 0
        
        if not self.driver:
            logger.error("Driver not initialized")
            return 0
            
        with self.driver.session() as session:
            for node in schema.nodes:
                # Create index on unique key
                if node.unique_key:
                    clean_property = self._clean_property_name(node.unique_key)
                    index_name = f"index_{node.node_type.lower()}_{clean_property}"
                    
                    cypher = f"""
                    CREATE INDEX {index_name} IF NOT EXISTS
                    FOR (n:{node.node_type})
                    ON (n.{clean_property})
                    """
                    
                    try:
                        session.run(cypher)
                        indexes_created += 1
                        logger.info(f"Created index: {index_name}")
                    except Exception as e:
                        logger.warning(f"Failed to create index {index_name}: {e}")
        
        return indexes_created
    
    def _ingest_nodes(self, schema: CSVSchemaExtraction, df: pd.DataFrame) -> int:
        """Ingest nodes based on the extracted schema"""
        total_nodes_created = 0
        
        if not self.driver:
            logger.error("Driver not initialized")
            return 0
        
        with self.driver.session() as session:
            for node_schema in schema.nodes:
                node_type = node_schema.node_type
                properties = list(node_schema.properties.keys())
                labels = ":".join(node_schema.labels)
                
                logger.info(f"Ingesting {node_type} nodes...")
                
                # Dynamic node extraction based on CSV structure
                node_data = self._extract_nodes_dynamic(df, node_schema)
                
                # Create nodes in batch
                nodes_created = self._create_nodes_batch(session, node_type, labels, node_data)
                total_nodes_created += nodes_created
                
                logger.info(f"Created {nodes_created} {node_type} nodes")
        
        return total_nodes_created
    
    def _extract_nodes_dynamic(self, df: pd.DataFrame, node_schema) -> List[Dict[str, Any]]:
        """Dynamically extract nodes based on CSV structure and schema"""
        nodes = []
        
        # Detect CSV type based on columns
        csv_columns = df.columns.tolist()
        
        if node_schema.node_type == "Patient":
            nodes = self._extract_patient_nodes(df)
        elif node_schema.node_type == "Finding":
            nodes = self._extract_finding_nodes_dynamic(df)
        elif node_schema.node_type == "Image":
            nodes = self._extract_image_nodes_dynamic(df)
        else:
            # Generic extraction for any other node types
            nodes = self._extract_generic_nodes(df, node_schema)
        
        return nodes
    
    def _extract_patient_nodes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract patient nodes from Data_Entry_2017.csv format - handles longitudinal data"""
        patient_nodes = []
        
        # Check if this CSV has patient data
        if 'Patient ID' not in df.columns:
            return patient_nodes
        
        # For longitudinal data, create a patient node for each unique (Patient ID, Age) combination
        # This allows us to track patients at different ages
        unique_patient_ages = df[['Patient ID', 'Patient Age', 'Patient Gender']].dropna().drop_duplicates()
        
        for _, row in unique_patient_ages.iterrows():
            patient_id = row['Patient ID']
            age = row['Patient Age']
            gender = row['Patient Gender']
            
            # Create unique ID combining patient and age for longitudinal tracking
            unique_id = f"{patient_id}_{age}"
            
            node_data = {
                'id': unique_id,
                'patient_id': str(patient_id),
                'age': int(age),
                'gender': str(gender),
                'original_patient_id': str(patient_id)  # Keep original for reference
            }
            
            patient_nodes.append(node_data)
            
        logger.info(f"Extracted {len(patient_nodes)} patient-age combinations (longitudinal data)")
        return patient_nodes
    
    def _extract_finding_nodes_dynamic(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract finding nodes from either CSV format"""
        finding_nodes = []
        
        # Determine finding column name
        finding_column = None
        if 'Finding Labels' in df.columns:
            finding_column = 'Finding Labels'
        elif 'Finding Label' in df.columns:
            finding_column = 'Finding Label'
        else:
            logger.warning("No finding column found in CSV")
            return finding_nodes
        
        # Get all findings (including pipe-separated ones)
        all_findings = set()
        
        for _, row in df.iterrows():
            if pd.notna(row[finding_column]):
                finding_value = str(row[finding_column])
                
                # Handle pipe-separated findings (e.g., "Cardiomegaly|Emphysema")
                if '|' in finding_value:
                    findings = finding_value.split('|')
                    for finding in findings:
                        all_findings.add(finding.strip())
                else:
                    all_findings.add(finding_value.strip())
        
        # Create finding nodes
        for finding in all_findings:
            if finding and finding != 'No Finding':  # Skip empty and "No Finding"
                # Count occurrences
                count = 0
                for _, row in df.iterrows():
                    if pd.notna(row[finding_column]):
                        finding_value = str(row[finding_column])
                        if finding in finding_value.split('|'):
                            count += 1
                
                node_data = {
                    'name': finding,
                    'finding_label': finding,
                    'occurrence_count': count
                }
                finding_nodes.append(node_data)
        
        logger.info(f"Extracted {len(finding_nodes)} unique findings")
        return finding_nodes
    
    def _extract_image_nodes_dynamic(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract image nodes from either CSV format"""
        image_nodes = []
        
        if 'Image Index' not in df.columns:
            return image_nodes
        
        for _, row in df.iterrows():
            if pd.notna(row['Image Index']):
                node_data = {
                    'id': str(row['Image Index']),
                    'image_index': str(row['Image Index'])
                }
                
                # Add bbox data if available (BBox_List_2017.csv format)
                if 'Bbox [x' in df.columns:
                    node_data['bbox_x'] = float(row['Bbox [x']) if pd.notna(row['Bbox [x']) else 0.0
                    if 'y' in df.columns:
                        node_data['bbox_y'] = float(row['y']) if pd.notna(row['y']) else 0.0
                    if 'w' in df.columns:
                        node_data['bbox_width'] = float(row['w']) if pd.notna(row['w']) else 0.0
                    if 'h' in df.columns:
                        node_data['bbox_height'] = float(row['h']) if pd.notna(row['h']) else 0.0
                
                # Add other image properties if available (Data_Entry_2017.csv format)
                if 'View Position' in df.columns and pd.notna(row['View Position']):
                    node_data['view_position'] = str(row['View Position'])
                    
                image_nodes.append(node_data)
        
        logger.info(f"Extracted {len(image_nodes)} image nodes")
        return image_nodes
    
    def _extract_generic_nodes(self, df: pd.DataFrame, node_schema) -> List[Dict[str, Any]]:
        """Generic node extraction for any node type"""
        nodes = []
        
        if node_schema.unique_key and node_schema.unique_key in df.columns:
            unique_values = df[node_schema.unique_key].dropna().unique()
            
            for value in unique_values:
                node_data = {self._clean_property_name(node_schema.unique_key): str(value)}
                
                # Add other properties
                for prop in node_schema.properties.keys():
                    if prop != node_schema.unique_key and prop in df.columns:
                        # Get the first non-null value for this unique key
                        prop_value = df[df[node_schema.unique_key] == value][prop].dropna().iloc[0] if not df[df[node_schema.unique_key] == value][prop].dropna().empty else None
                        if prop_value is not None:
                            node_data[self._clean_property_name(prop)] = str(prop_value)
                
                nodes.append(node_data)
        
        return nodes
    
    def _create_nodes_batch(self, session, node_type: str, labels: str, node_data: List[Dict[str, Any]]) -> int:
        """Create nodes in batch for better performance"""
        if not node_data:
            return 0
        
        cypher = f"""
        UNWIND $nodes AS nodeData
        CREATE (n:{labels})
        SET n += nodeData
        """
        
        try:
            result = session.run(cypher, nodes=node_data)
            return len(node_data)
        except Exception as e:
            logger.error(f"Failed to create {node_type} nodes: {e}")
            return 0
    
    def _ingest_relationships(self, schema: CSVSchemaExtraction, df: pd.DataFrame) -> int:
        """Ingest relationships based on the extracted schema"""
        total_relationships_created = 0
        
        if not self.driver:
            logger.error("Driver not initialized")
            return 0
        
        with self.driver.session() as session:
            # Always create Patient-Finding relationships if patient data exists
            if 'Patient ID' in df.columns:
                patient_finding_rels = self._create_patient_finding_relationships(session, df)
                total_relationships_created += patient_finding_rels
                logger.info(f"Created {patient_finding_rels} Patient-Finding relationships")
            
            # Create Image-Finding relationships
            if 'Image Index' in df.columns:
                image_finding_rels = self._create_image_finding_relationships_dynamic(session, df)
                total_relationships_created += image_finding_rels
                logger.info(f"Created {image_finding_rels} Image-Finding relationships")
            
            # Create any additional relationships from schema
            for rel_schema in schema.relationships:
                if rel_schema.relationship_type not in ['HAS_FINDING']:  # Skip already created
                    relationships_created = self._create_generic_relationships(session, rel_schema, df)
                    total_relationships_created += relationships_created
                    logger.info(f"Created {relationships_created} {rel_schema.relationship_type} relationships")
        
        return total_relationships_created
    
    def _create_patient_finding_relationships(self, session, df: pd.DataFrame) -> int:
        """Create Patient-Finding relationships that GraphRAG agent expects - handles longitudinal data"""
        if 'Patient ID' not in df.columns:
            return 0
            
        # Determine finding column name
        finding_column = None
        if 'Finding Labels' in df.columns:
            finding_column = 'Finding Labels'
        elif 'Finding Label' in df.columns:
            finding_column = 'Finding Label'
        else:
            return 0
        
        cypher = """
        UNWIND $relationships AS rel
        MATCH (p:Patient {id: rel.patient_id})
        MATCH (f:Finding {name: rel.finding_name})
        MERGE (p)-[:HAS_FINDING]->(f)
        """
        
        relationships_data = []
        
        for _, row in df.iterrows():
            if pd.notna(row['Patient ID']) and pd.notna(row['Patient Age']) and pd.notna(row[finding_column]):
                patient_id = str(row['Patient ID'])
                age = int(row['Patient Age'])
                unique_patient_id = f"{patient_id}_{age}"  # Use the same format as patient nodes
                finding_value = str(row[finding_column])
                
                # Handle pipe-separated findings
                if '|' in finding_value:
                    findings = finding_value.split('|')
                    for finding in findings:
                        finding = finding.strip()
                        if finding and finding != 'No Finding':
                            relationships_data.append({
                                'patient_id': unique_patient_id,
                                'finding_name': finding
                            })
                else:
                    finding = finding_value.strip()
                    if finding and finding != 'No Finding':
                        relationships_data.append({
                            'patient_id': unique_patient_id,
                            'finding_name': finding
                        })
        
        # Remove duplicates
        unique_relationships = []
        seen = set()
        for rel in relationships_data:
            key = (rel['patient_id'], rel['finding_name'])
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        try:
            session.run(cypher, relationships=unique_relationships)
            return len(unique_relationships)
        except Exception as e:
            logger.error(f"Failed to create patient-finding relationships: {e}")
            return 0
    
    def _create_image_finding_relationships_dynamic(self, session, df: pd.DataFrame) -> int:
        """Create Image-Finding relationships for both CSV formats"""
        if 'Image Index' not in df.columns:
            return 0
            
        # Determine finding column name
        finding_column = None
        if 'Finding Labels' in df.columns:
            finding_column = 'Finding Labels'
        elif 'Finding Label' in df.columns:
            finding_column = 'Finding Label'
        else:
            return 0
        
        # Check if this is BBox format (has bbox coordinates)
        has_bbox = 'Bbox [x' in df.columns
        
        if has_bbox:
            cypher = """
            UNWIND $relationships AS rel
            MATCH (img:Image {id: rel.image_index})
            MATCH (finding:Finding {name: rel.finding_name})
            MERGE (img)-[:HAS_FINDING {
                bbox_x: rel.bbox_x,
                bbox_y: rel.bbox_y, 
                bbox_width: rel.bbox_width,
                bbox_height: rel.bbox_height
            }]->(finding)
            """
        else:
            cypher = """
            UNWIND $relationships AS rel
            MATCH (img:Image {id: rel.image_index})
            MATCH (finding:Finding {name: rel.finding_name})
            MERGE (img)-[:HAS_FINDING]->(finding)
            """
        
        relationships_data = []
        
        for _, row in df.iterrows():
            if pd.notna(row['Image Index']) and pd.notna(row[finding_column]):
                image_index = str(row['Image Index'])
                finding_value = str(row[finding_column])
                
                # Handle pipe-separated findings
                if '|' in finding_value:
                    findings = finding_value.split('|')
                    for finding in findings:
                        finding = finding.strip()
                        if finding and finding != 'No Finding':
                            rel_data = {
                                'image_index': image_index,
                                'finding_name': finding
                            }
                            
                            if has_bbox:
                                rel_data.update({
                                    'bbox_x': float(row['Bbox [x']) if pd.notna(row['Bbox [x']) else 0.0,
                                    'bbox_y': float(row['y']) if pd.notna(row['y']) else 0.0,
                                    'bbox_width': float(row['w']) if pd.notna(row['w']) else 0.0,
                                    'bbox_height': float(row['h']) if pd.notna(row['h']) else 0.0,
                                })
                            
                            relationships_data.append(rel_data)
                else:
                    finding = finding_value.strip()
                    if finding and finding != 'No Finding':
                        rel_data = {
                            'image_index': image_index,
                            'finding_name': finding
                        }
                        
                        if has_bbox:
                            rel_data.update({
                                'bbox_x': float(row['Bbox [x']) if pd.notna(row['Bbox [x']) else 0.0,
                                'bbox_y': float(row['y']) if pd.notna(row['y']) else 0.0,
                                'bbox_width': float(row['w']) if pd.notna(row['w']) else 0.0,
                                'bbox_height': float(row['h']) if pd.notna(row['h']) else 0.0,
                            })
                        
                        relationships_data.append(rel_data)
        
        try:
            session.run(cypher, relationships=relationships_data)
            return len(relationships_data)
        except Exception as e:
            logger.error(f"Failed to create image-finding relationships: {e}")
            return 0
    
    def _create_generic_relationships(self, session, rel_schema, df: pd.DataFrame) -> int:
        """Create generic relationships between nodes"""
        # This would be implemented based on specific relationship requirements
        logger.info(f"Generic relationship creation not yet implemented for {rel_schema.relationship_type}")
        return 0
    
    def _clean_property_name(self, prop_name: str) -> str:
        """Clean property names for Neo4j compatibility"""
        # Replace spaces and special characters
        clean_name = prop_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '_').replace('#', 'number')
        # Remove any remaining special characters
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        return clean_name
    
    def query_graph(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        if not self.driver:
            logger.error("Driver not initialized")
            return []
            
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [record.data() for record in result]
    
    def get_graph_statistics(self) -> Dict[str, int]:
        """Get basic statistics about the graph"""
        stats = {}
        
        if not self.driver:
            logger.error("Driver not initialized")
            return stats
        
        with self.driver.session() as session:
            # Count nodes by label
            result = session.run("MATCH (n) RETURN labels(n) AS labels, count(n) AS count")
            node_counts = {}
            for record in result:
                labels = record['labels']
                count = record['count']
                for label in labels:
                    node_counts[label] = node_counts.get(label, 0) + count
            stats['nodes'] = node_counts
            
            # Count relationships by type
            result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count")
            rel_counts = {record['type']: record['count'] for record in result}
            stats['relationships'] = rel_counts
        
        return stats
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Example usage"""
    # Use the correct path to Data_Entry_2017.csv which has patient information
    csv_file = r"c:\Users\udaybhanu.dutta\HCL work\Gen AI\Agents\LangGraph_projects\RAG-agents\doc-ingestion\Data_Entry_2017.csv"
    
    # Initialize Neo4j ingestor using keyvault manager (falls back to .env.dev when keyvault is disabled)
    with Neo4jCSVIngestor() as ingestor:
        
        # Ingest CSV to Neo4j
        result = ingestor.ingest_csv_to_neo4j(csv_file, clear_existing=True)
        
        print(f"Ingestion Result:")
        print(f"  Success: {result.success}")
        print(f"  Nodes Created: {result.nodes_created}")
        print(f"  Relationships Created: {result.relationships_created}")
        print(f"  Execution Time: {result.execution_time:.2f}s")
        
        if result.errors:
            print(f"  Errors: {result.errors}")
        
        # Get statistics
        stats = ingestor.get_graph_statistics()
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {stats['nodes']}")
        print(f"  Relationships: {stats['relationships']}")
        
        # Example queries that match what GraphRAG agent expects
        print(f"\nExample Queries:")
        
        # Test the exact query that was failing
        query1 = """
        MATCH (p:Patient)-[:HAS_FINDING]->(f:Finding) 
        WHERE p.age < 40 
        RETURN p.id as patient_id, p.age as age, p.gender as gender, f.name as finding
        LIMIT 5
        """
        results1 = ingestor.query_graph(query1)
        print(f"Patients under 40 with findings: {len(results1)} found")
        for result in results1[:3]:
            print(f"  Patient {result['patient_id']}: {result['age']}yo {result['gender']}, finding: {result['finding']}")
        
        # Count patients by gender
        query2 = """
        MATCH (p:Patient)
        RETURN p.gender, count(*) as count
        ORDER BY count DESC
        """
        results2 = ingestor.query_graph(query2)
        print(f"Patients by gender: {results2}")
        
        # Count findings by type
        query3 = """
        MATCH (f:Finding)
        RETURN f.name, f.occurrence_count
        ORDER BY f.occurrence_count DESC
        LIMIT 10
        """
        results3 = ingestor.query_graph(query3)
        print(f"Top findings: {results3}")


if __name__ == "__main__":
    main()
