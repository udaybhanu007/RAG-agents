
import json
from neo4j import GraphDatabase

# Neo4j AuraDB connection details (consider moving to config/env)
uri = "neo4j+ssc://2b631765.databases.neo4j.io"
username = "neo4j"
password = "Fn5EIxDQPVVd8lvoOQgcYk8nwPKyvMTk5eHnRhI3K34"

def ingest_to_neo4j(data_patient: dict, data_bbox: dict, uri: str = uri, username: str = username, password: str = password):
    """
    Ingest patient and bbox data (already extracted) into Neo4j.
    """
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        try:
            # Batch ingest for patient-medical JSON
            patient_data = data_patient.get("patient_medical_relationships", {}).get("patient_findings", {})
            patient_batch = []
            for patient_id, details in patient_data.items():
                findings = details.get("unique_findings", [])
                finding_freq = details.get("finding_frequency", {})
                age_range = details.get("age_range", [0, 0])
                gender = details.get("gender", "")
                total_images = details.get("total_images", 0)
                for finding in findings:
                    patient_batch.append({
                        "patient_id": patient_id,
                        "finding": finding,
                        "frequency": finding_freq.get(finding, 1),
                        "age_min": age_range[0],
                        "age_max": age_range[1],
                        "gender": gender,
                        "total_images": total_images
                    })

            BATCH_SIZE = 1000
            for i in range(0, len(patient_batch), BATCH_SIZE):
                batch = patient_batch[i:i+BATCH_SIZE]
                session.run('''
                    UNWIND $batch AS row
                    MERGE (p:Patient {id: row.patient_id})
                    SET p.age_min = row.age_min,
                        p.age_max = row.age_max,
                        p.gender = row.gender,
                        p.total_images = row.total_images
                    MERGE (f:Finding {name: row.finding})
                    MERGE (p)-[r:HAS_FINDING]->(f)
                    SET r.frequency = row.frequency
                ''', batch=batch)
                print(f"[Patient] Batch {i//BATCH_SIZE+1} ingested ({min(i+BATCH_SIZE, len(patient_batch))}/{len(patient_batch)})")

            print("✅ Patient-medical ingestion complete.")

            # Batch ingest for bbox JSON
            bbox_details = data_bbox.get("bbox_details", {})
            bbox_batch = []
            for bbox_id, bbox_info in bbox_details.items():
                parts = bbox_id.split("_")
                for i, part in enumerate(parts):
                    if part.endswith(".png"):
                        image_id = "_".join(parts[:i+1])
                        finding = parts[i+1]
                        x, y, width, height = map(float, parts[i+2:i+6])
                        break
                else:
                    image_id = parts[0]
                    finding = parts[1]
                    x, y, width, height = map(float, parts[2:6])
                bbox_batch.append({
                    "image_id": image_id,
                    "finding": finding,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                })

            for i in range(0, len(bbox_batch), BATCH_SIZE):
                batch = bbox_batch[i:i+BATCH_SIZE]
                session.run('''
                    UNWIND $batch AS row
                    MERGE (i:Image {id: row.image_id})
                    MERGE (f:Finding {name: row.finding})
                    MERGE (i)-[:HAS_FINDING]->(f)
                    MERGE (b:BBox {
                        x: row.x, y: row.y, width: row.width, height: row.height,
                        image_id: row.image_id, finding: row.finding
                    })
                    MERGE (i)-[:HAS_BBOX]->(b)
                    MERGE (b)-[:DETECTS]->(f)
                ''', batch=batch)
                print(f"[BBox] Batch {i//BATCH_SIZE+1} ingested ({min(i+BATCH_SIZE, len(bbox_batch))}/{len(bbox_batch)})")

            print("✅ BBox relationships ingestion complete.")

        except Exception as e:
            print(f"❌ Ingestion failed: {e}")

# For backward compatibility, keep the script runnable directly (optional)
if __name__ == "__main__":
    # Example: load from files and call the function
    json_path_patient = r"D:\Softwares\Neo4j-poc\Neo4j_Ingestion\output_folder\Data_Entry_2017_entity_relationships.json"
    json_path_bbox = r"D:\Softwares\Neo4j-poc\Neo4j_Ingestion\output_folder\BBox_List_2017_entity_relationships.json"
    with open(json_path_patient, 'r', encoding='utf-8') as f:
        data_patient = json.load(f)
    with open(json_path_bbox, 'r', encoding='utf-8') as f:
        data_bbox = json.load(f)
    ingest_to_neo4j(data_patient, data_bbox)

def ingest_patient_findings(tx, patient_id, findings, age_range, gender, total_images, finding_freq):
    tx.run("""
        MERGE (p:Patient {id: $patient_id})
        SET p.age_min = $age_min,
            p.age_max = $age_max,
            p.gender = $gender,
            p.total_images = $total_images
    """, patient_id=patient_id, age_min=age_range[0], age_max=age_range[1],
         gender=gender, total_images=total_images)

    for finding in findings:
        frequency = finding_freq.get(finding, 1)
        tx.run("""
            MERGE (f:Finding {name: $finding})
            MERGE (p:Patient {id: $patient_id})
            MERGE (p)-[r:HAS_FINDING]->(f)
            SET r.frequency = $frequency
        """, finding=finding, patient_id=patient_id, frequency=frequency)

def ingest_bbox_relationship(tx, image_id, finding, bbox):
    tx.run("""
        MERGE (i:Image {id: $image_id})
        MERGE (f:Finding {name: $finding})
        MERGE (i)-[:HAS_FINDING]->(f)
        MERGE (b:BBox {
            x: $x, y: $y, width: $width, height: $height,
            image_id: $image_id, finding: $finding
        })
        MERGE (i)-[:HAS_BBOX]->(b)
        MERGE (b)-[:DETECTS]->(f)
    """, image_id=image_id, finding=finding,
         x=bbox['x'], y=bbox['y'], width=bbox['width'], height=bbox['height'])



