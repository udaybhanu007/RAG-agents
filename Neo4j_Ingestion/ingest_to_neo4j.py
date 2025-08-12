"""
Medical Image Data Ingestion Script for Neo4j
==============================================

This script processes medical imaging CSV files and creates a structured knowledge graph
with proper schema definition and simplified logic.

SCHEMA DEFINITION:
=================

NODES:
------
1. Patient: Represents a patient with medical images
   Properties: id, age, gender, total_images

2. Image: Represents a medical image (X-ray)
   Properties: id, view_position, followup_number, original_width, original_height, 
              pixel_spacing_x, pixel_spacing_y

3. Finding: Represents a medical finding/condition
   Properties: name

4. BoundingBox: Represents bounding box coordinates for findings in images
   Properties: id, x, y, width, height, area

RELATIONSHIPS:
--------------
1. Patient -[HAS_IMAGE]-> Image
2. Image -[HAS_FINDING]-> Finding  
3. Patient -[HAS_FINDING {frequency}]-> Finding
4. Image -[HAS_BBOX]-> BoundingBox
5. BoundingBox -[DETECTS]-> Finding

"""

import pandas as pd
import json
from neo4j import GraphDatabase
from collections import defaultdict

# Neo4j AuraDB connection details
uri = "neo4j+ssc://2b631765.databases.neo4j.io"
username = "neo4j"
password = "Fn5EIxDQPVVd8lvoOQgcYk8nwPKyvMTk5eHnRhI3K34"

# CSV file paths
data_entry_csv = r"D:\Softwares\Neo4j-poc\doc-ingestion\Data_Entry_2017.csv"
bbox_csv = r"D:\Softwares\Neo4j-poc\doc-ingestion\BBox_List_2017.csv"

class MedicalImageGraphIngestion:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.batch_size = 1000
    
    def process_data_entry_csv(self, csv_path):
        """Process Data_Entry_2017.csv and extract structured data"""
        print("📊 Processing Data Entry CSV...")
        
        # Read CSV with proper handling
        df = pd.read_csv(csv_path)
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        patients = {}
        images = []
        findings = set()
        patient_findings = defaultdict(lambda: defaultdict(int))
        
        for _, row in df.iterrows():
            try:
                # Extract basic data using actual column names
                image_id = str(row['Image Index']).strip()
                patient_id = str(row['Patient ID']).strip()
                finding_labels = str(row['Finding Labels']).strip()
                patient_age = int(row['Patient Age']) if pd.notna(row['Patient Age']) else None
                patient_gender = str(row['Patient Gender']).strip() if pd.notna(row['Patient Gender']) else None
                view_position = str(row['View Position']).strip() if pd.notna(row['View Position']) else None
                followup_num = int(row['Follow-up #']) if pd.notna(row['Follow-up #']) else 0
                
                # Extract image dimensions from split columns
                # Columns: 'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]'
                original_width = int(row['OriginalImage[Width']) if pd.notna(row['OriginalImage[Width']) else None
                original_height = int(row['Height]']) if pd.notna(row['Height]']) else None
                pixel_spacing_x = float(row['OriginalImagePixelSpacing[x']) if pd.notna(row['OriginalImagePixelSpacing[x']) else None
                pixel_spacing_y = float(row['y]']) if pd.notna(row['y]']) else None
                
                # Store patient data
                if patient_id not in patients:
                    patients[patient_id] = {
                        'id': patient_id,
                        'age': patient_age,
                        'gender': patient_gender,
                        'total_images': 0
                    }
                
                patients[patient_id]['total_images'] += 1
                
                # Store image data
                images.append({
                    'id': image_id,
                    'patient_id': patient_id,
                    'view_position': view_position,
                    'followup_number': followup_num,
                    'original_width': original_width,
                    'original_height': original_height,
                    'pixel_spacing_x': pixel_spacing_x,
                    'pixel_spacing_y': pixel_spacing_y
                })
                
                # Process findings
                if finding_labels and finding_labels not in ['nan', 'No Finding']:
                    finding_list = [f.strip() for f in finding_labels.split('|')]
                    for finding in finding_list:
                        if finding:
                            findings.add(finding)
                            patient_findings[patient_id][finding] += 1
                            
            except Exception as e:
                print(f"⚠️ Error processing row: {e}")
                continue
        
        print(f"✅ Processed {len(patients)} patients, {len(images)} images, {len(findings)} findings")
        return patients, images, list(findings), dict(patient_findings)
    
    def process_bbox_csv(self, csv_path):
        """Process BBox_List_2017.csv and extract bounding box data"""
        print("📦 Processing BBox CSV...")
        
        df = pd.read_csv(csv_path)
        
        bboxes = []
        
        for _, row in df.iterrows():
            try:
                image_id = str(row['Image Index']).strip()
                finding = str(row['Finding Label']).strip()
                
                # Extract bbox coordinates from correct columns
                # Columns: 'Bbox [x', 'y', 'w', 'h]'
                bbox_x = float(row['Bbox [x']) if pd.notna(row['Bbox [x']) else 0.0
                bbox_y = float(row['y']) if pd.notna(row['y']) else 0.0
                bbox_w = float(row['w']) if pd.notna(row['w']) else 0.0
                bbox_h = float(row['h]']) if pd.notna(row['h]']) else 0.0
                
                bbox_id = f"{image_id}_{finding}_{int(bbox_x)}_{int(bbox_y)}"
                
                bboxes.append({
                    'id': bbox_id,
                    'image_id': image_id,
                    'finding': finding,
                    'x': bbox_x,
                    'y': bbox_y,
                    'width': bbox_w,
                    'height': bbox_h,
                    'area': bbox_w * bbox_h
                })
                
            except Exception as e:
                print(f"⚠️ Error processing bbox row: {e}")
                continue
        
        print(f"✅ Processed {len(bboxes)} bounding boxes")
        return bboxes
    
    def clear_database(self):
        """Clear all existing data"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️ Database cleared")
    
    def create_constraints_and_indexes(self):
        """Create database constraints and indexes for better performance"""
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT image_id IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE", 
                "CREATE CONSTRAINT finding_name IF NOT EXISTS FOR (f:Finding) REQUIRE f.name IS UNIQUE",
                "CREATE CONSTRAINT bbox_id IF NOT EXISTS FOR (b:BoundingBox) REQUIRE b.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Constraint already exists or error: {e}")
            
            print("📋 Constraints and indexes created")
    
    def ingest_patients(self, patients):
        """Ingest patient nodes"""
        print("👥 Ingesting patients...")
        
        with self.driver.session() as session:
            patient_list = list(patients.values())
            
            for i in range(0, len(patient_list), self.batch_size):
                batch = patient_list[i:i + self.batch_size]
                session.run("""
                    UNWIND $batch AS patient
                    CREATE (p:Patient {
                        id: patient.id,
                        age: patient.age,
                        gender: patient.gender,
                        total_images: patient.total_images
                    })
                """, batch=batch)
                
                print(f"  Patients batch {i//self.batch_size + 1}: {len(batch)} patients")
    
    def ingest_images(self, images):
        """Ingest image nodes and patient-image relationships"""
        print("🖼️ Ingesting images...")
        
        with self.driver.session() as session:
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size]
                session.run("""
                    UNWIND $batch AS image
                    CREATE (i:Image {
                        id: image.id,
                        view_position: image.view_position,
                        followup_number: image.followup_number,
                        original_width: image.original_width,
                        original_height: image.original_height,
                        pixel_spacing_x: image.pixel_spacing_x,
                        pixel_spacing_y: image.pixel_spacing_y
                    })
                    WITH i, image
                    MATCH (p:Patient {id: image.patient_id})
                    CREATE (p)-[:HAS_IMAGE]->(i)
                """, batch=batch)
                
                print(f"  Images batch {i//self.batch_size + 1}: {len(batch)} images")
    
    def ingest_findings(self, findings):
        """Ingest finding nodes"""
        print("🔍 Ingesting findings...")
        
        with self.driver.session() as session:
            finding_batch = [{'name': finding} for finding in findings]
            
            for i in range(0, len(finding_batch), self.batch_size):
                batch = finding_batch[i:i + self.batch_size]
                session.run("""
                    UNWIND $batch AS finding
                    CREATE (f:Finding {name: finding.name})
                """, batch=batch)
                
                print(f"  Findings batch {i//self.batch_size + 1}: {len(batch)} findings")
    
    def ingest_bboxes(self, bboxes):
        """Ingest bounding box nodes and relationships"""
        print("📦 Ingesting bounding boxes...")
        
        with self.driver.session() as session:
            for i in range(0, len(bboxes), self.batch_size):
                batch = bboxes[i:i + self.batch_size]
                session.run("""
                    UNWIND $batch AS bbox
                    CREATE (b:BoundingBox {
                        id: bbox.id,
                        x: bbox.x,
                        y: bbox.y,
                        width: bbox.width,
                        height: bbox.height,
                        area: bbox.area
                    })
                    WITH b, bbox
                    MATCH (i:Image {id: bbox.image_id})
                    MATCH (f:Finding {name: bbox.finding})
                    CREATE (i)-[:HAS_BBOX]->(b)
                    CREATE (b)-[:DETECTS]->(f)
                """, batch=batch)
                
                print(f"  BBox batch {i//self.batch_size + 1}: {len(batch)} bounding boxes")
    
    def create_image_finding_relationships(self, images):
        """Create image-finding relationships based on image data"""
        print("🔗 Creating image-finding relationships...")
        
        with self.driver.session() as session:
            # This will be derived from the CSV data during image processing
            session.run("""
                MATCH (i:Image)-[:HAS_BBOX]->(b:BoundingBox)-[:DETECTS]->(f:Finding)
                MERGE (i)-[:HAS_FINDING]->(f)
            """)
            
            # Also create relationships for findings mentioned in the CSV but without bboxes
            session.run("""
                MATCH (i:Image)<-[:HAS_IMAGE]-(p:Patient)
                MATCH (b:BoundingBox {image_id: i.id})-[:DETECTS]->(f:Finding)
                MERGE (i)-[:HAS_FINDING]->(f)
            """)
    
    def create_patient_finding_relationships(self, patient_findings):
        """Create patient-finding relationships with frequency"""
        print("👤 Creating patient-finding relationships...")
        
        relationships = []
        for patient_id, findings in patient_findings.items():
            for finding, frequency in findings.items():
                relationships.append({
                    'patient_id': patient_id,
                    'finding': finding,
                    'frequency': frequency
                })
        
        with self.driver.session() as session:
            for i in range(0, len(relationships), self.batch_size):
                batch = relationships[i:i + self.batch_size]
                session.run("""
                    UNWIND $batch AS rel
                    MATCH (p:Patient {id: rel.patient_id})
                    MATCH (f:Finding {name: rel.finding})
                    CREATE (p)-[:HAS_FINDING {frequency: rel.frequency}]->(f)
                """, batch=batch)
                
                print(f"  Patient-Finding batch {i//self.batch_size + 1}: {len(batch)} relationships")
    
    def print_summary(self):
        """Print database summary statistics"""
        with self.driver.session() as session:
            # Count nodes
            result = session.run("MATCH (p:Patient) RETURN count(p) as count")
            patient_count = result.single()["count"]
            
            result = session.run("MATCH (i:Image) RETURN count(i) as count")
            image_count = result.single()["count"]
            
            result = session.run("MATCH (f:Finding) RETURN count(f) as count")
            finding_count = result.single()["count"]
            
            result = session.run("MATCH (b:BoundingBox) RETURN count(b) as count")
            bbox_count = result.single()["count"]
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            relationship_count = result.single()["count"]
            
            print("\n" + "="*50)
            print("📊 DATABASE SUMMARY")
            print("="*50)
            print(f"👥 Patients: {patient_count}")
            print(f"🖼️ Images: {image_count}")
            print(f"🔍 Findings: {finding_count}")
            print(f"📦 Bounding Boxes: {bbox_count}")
            print(f"🔗 Total Relationships: {relationship_count}")
            print("="*50)
    
    def close(self):
        self.driver.close()

def main():
    # Initialize ingestion class
    ingestion = MedicalImageGraphIngestion(uri, username, password)
    
    try:
        # Clear existing data
        ingestion.clear_database()
        
        # Create constraints and indexes
        ingestion.create_constraints_and_indexes()
        
        # Process CSV files
        patients, images, findings, patient_findings = ingestion.process_data_entry_csv(data_entry_csv)
        bboxes = ingestion.process_bbox_csv(bbox_csv)
        
        # Ingest data in order
        ingestion.ingest_patients(patients)
        ingestion.ingest_findings(findings)
        ingestion.ingest_images(images)
        ingestion.ingest_bboxes(bboxes)
        
        # Create relationships
        ingestion.create_image_finding_relationships(images)
        ingestion.create_patient_finding_relationships(patient_findings)
        
        # Print summary
        ingestion.print_summary()
        
        print("\n✅ Medical image data ingestion completed successfully!")
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        ingestion.close()

if __name__ == "__main__":
    main()