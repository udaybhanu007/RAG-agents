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
                
                # Get unique values for this node type
                if node_type == "Image":
                    # For images, each row is a unique image
                    node_data = self._extract_image_nodes(df)
                elif node_type == "Finding":
                    # For findings, get unique finding labels
                    node_data = self._extract_finding_nodes(df)
                else:
                    # Generic node extraction
                    node_data = self._extract_generic_nodes(df, node_schema)
                
                # Create nodes in batch
                nodes_created = self._create_nodes_batch(session, node_type, labels, node_data)
                total_nodes_created += nodes_created
                
                logger.info(f"Created {nodes_created} {node_type} nodes")
        
        return total_nodes_created
    
    def _extract_image_nodes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract image node data from DataFrame"""
        image_nodes = []
        
        for _, row in df.iterrows():
            if pd.notna(row['Image Index']):
                node_data = {
                    'image_index': str(row['Image Index']),
                    'bbox_x': float(row['Bbox [x']) if pd.notna(row['Bbox [x']) else 0.0,
                    'bbox_y': float(row['y']) if pd.notna(row['y']) else 0.0,
                    'bbox_width': float(row['w']) if pd.notna(row['w']) else 0.0,
                    'bbox_height': float(row['h]']) if pd.notna(row['h]']) else 0.0,
                }
                image_nodes.append(node_data)
        
        return image_nodes
    
    def _extract_finding_nodes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract finding node data from DataFrame"""
        # Get unique findings
        unique_findings = df['Finding Label'].dropna().unique()
        
        finding_nodes = []
        for finding in unique_findings:
            node_data = {
                'finding_label': str(finding),
                'occurrence_count': int(df[df['Finding Label'] == finding].shape[0])
            }
            finding_nodes.append(node_data)
        
        return finding_nodes
    
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
            for rel_schema in schema.relationships:
                logger.info(f"Creating {rel_schema.relationship_type} relationships...")
                
                # Create relationships based on the schema
                if rel_schema.source_node == "Image" and rel_schema.target_node == "Finding":
                    relationships_created = self._create_image_finding_relationships(session, df)
                else:
                    relationships_created = self._create_generic_relationships(session, rel_schema, df)
                
                total_relationships_created += relationships_created
                logger.info(f"Created {relationships_created} {rel_schema.relationship_type} relationships")
        
        return total_relationships_created
    
    def _create_image_finding_relationships(self, session, df: pd.DataFrame) -> int:
        """Create relationships between images and findings"""
        cypher = """
        UNWIND $relationships AS rel
        MATCH (img:Image {image_index: rel.image_index})
        MATCH (finding:Finding {finding_label: rel.finding_label})
        CREATE (img)-[:HAS_FINDING {
            bbox_x: rel.bbox_x,
            bbox_y: rel.bbox_y, 
            bbox_width: rel.bbox_width,
            bbox_height: rel.bbox_height
        }]->(finding)
        """
        
        relationships_data = []
        for _, row in df.iterrows():
            if pd.notna(row['Image Index']) and pd.notna(row['Finding Label']):
                rel_data = {
                    'image_index': str(row['Image Index']),
                    'finding_label': str(row['Finding Label']),
                    'bbox_x': float(row['Bbox [x']) if pd.notna(row['Bbox [x']) else 0.0,
                    'bbox_y': float(row['y']) if pd.notna(row['y']) else 0.0,
                    'bbox_width': float(row['w']) if pd.notna(row['w']) else 0.0,
                    'bbox_height': float(row['h]']) if pd.notna(row['h]']) else 0.0,
                }
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
    csv_file = "doc-ingestion/BBox_List_2017.csv"
    
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
        
        # Example queries
        print(f"\nExample Queries:")
        
        # Find images with Atelectasis
        query1 = """
        MATCH (img:Image)-[:HAS_FINDING]->(f:Finding {finding_label: 'Atelectasis'})
        RETURN img.image_index, f.finding_label
        LIMIT 5
        """
        results1 = ingestor.query_graph(query1)
        print(f"Images with Atelectasis: {len(results1)} found")
        
        # Count findings by type
        query2 = """
        MATCH (f:Finding)
        RETURN f.finding_label, f.occurrence_count
        ORDER BY f.occurrence_count DESC
        """
        results2 = ingestor.query_graph(query2)
        print(f"Finding types: {len(results2)} different findings")


if __name__ == "__main__":
    main()
