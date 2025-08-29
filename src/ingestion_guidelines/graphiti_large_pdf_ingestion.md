# Graphiti Large PDF Ingestion Strategy - Rate Limiting Solutions

## Root Cause Analysis

Your rate limiting error occurs because **Graphiti makes multiple API calls per chunk**:

1. **Entity Extraction**: LLM call to identify entities in each chunk
2. **Relationship Extraction**: LLM call to identify relationships 
3. **Entity Resolution**: LLM calls to resolve duplicate entities across chunks
4. **Embedding Generation**: API calls for semantic vectors
5. **Cross-Reference Resolution**: Additional LLM calls for chunk connections

**Result**: Each "chunk" triggers 5-10+ API calls, quickly overwhelming rate limits even with conservative chunking.

## Immediate Solutions

### Solution 1: Environment Variable Configuration

By default, SEMAPHORE_LIMIT is set to 10 concurrent operations to help prevent 429 rate limit errors from your LLM provider. If you encounter such errors, try lowering this value.

```bash
# Set these environment variables BEFORE running your ingestion
export SEMAPHORE_LIMIT=1          # Reduce from 10 to 1 (sequential processing)
export GRAPHITI_TELEMETRY_ENABLED=false  # Reduce API overhead

# Optional: Add delays between operations
export GRAPHITI_API_DELAY=2000    # 2 second delay between API calls (if supported)
```

```python
# Or set in your Python code before importing Graphiti
import os
os.environ['SEMAPHORE_LIMIT'] = '1'
os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'

from graphiti_core import Graphiti
```

### Solution 2: Strategic Document Chunking

**Current Issue**: Your 8 chunks × 5-10 API calls = 40-80 API calls in rapid succession

**Better Approach**: Larger, more meaningful chunks

```python
class OptimizedGraphitiIngestion:
    def __init__(self):
        # Set conservative concurrency
        os.environ['SEMAPHORE_LIMIT'] = '1'
        
        self.graphiti = Graphiti(
            graph_driver=your_driver,
            llm_client=your_llm_client
        )
    
    def create_semantic_chunks(self, pdf_content, max_chunk_size=8000):
        """
        Create fewer, larger, semantically meaningful chunks
        """
        # Split by major sections rather than arbitrary length
        sections = self.split_by_sections(pdf_content)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for section in sections:
            if current_size + len(section) < max_chunk_size:
                current_chunk += section
                current_size += len(section)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = section
                current_size = len(section)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def split_by_sections(self, content):
        """
        Split by document structure rather than arbitrary length
        """
        # For your NIH document, split by major sections
        section_markers = [
            "Background & Motivation:",
            "Details:",
            "Contents:",
            "Limitations:",
            "Acknowledgement:",
            "Reference:",
            "co-occurrence matrix"
        ]
        
        sections = []
        current_section = ""
        
        for line in content.split('\n'):
            if any(marker in line for marker in section_markers):
                if current_section:
                    sections.append(current_section)
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        if current_section:
            sections.append(current_section)
            
        return sections
```

### Solution 3: Batch Processing with Delays

```python
import asyncio
from datetime import datetime, timedelta

class RateLimitedIngestion:
    def __init__(self, requests_per_minute=30):
        self.rpm_limit = requests_per_minute
        self.request_times = []
        
    async def controlled_add_episode(self, graphiti_instance, **episode_kwargs):
        """
        Add episode with rate limiting control
        """
        # Clean old timestamps
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        # Check if we need to wait
        if len(self.request_times) >= self.rpm_limit:
            sleep_time = 60 - (now - self.request_times[0]).total_seconds()
            if sleep_time > 0:
                print(f"⏱️ Rate limiting: Sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        # Record this request time
        self.request_times.append(now)
        
        # Make the actual API call
        try:
            result = await graphiti_instance.add_episode(**episode_kwargs)
            print(f"✅ Successfully added episode: {episode_kwargs.get('name', 'unnamed')}")
            return result
        except Exception as e:
            if "rate limit" in str(e).lower():
                print(f"⚠️ Hit rate limit, waiting 60s before retry...")
                await asyncio.sleep(60)
                return await graphiti_instance.add_episode(**episode_kwargs)
            else:
                raise e

    async def ingest_large_pdf(self, pdf_content, document_name):
        """
        Ingest large PDF with comprehensive rate limiting
        """
        # Create fewer, larger chunks
        chunks = self.create_semantic_chunks(pdf_content, max_chunk_size=12000)
        
        print(f"📄 Processing {document_name} in {len(chunks)} semantic chunks")
        
        successful_chunks = 0
        failed_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                print(f"🔄 Processing chunk {i+1}/{len(chunks)}")
                
                await self.controlled_add_episode(
                    self.graphiti,
                    name=f"{document_name}_section_{i+1}",
                    episode_body=chunk,
                    source_description=f"Section {i+1} of {document_name}",
                    metadata={
                        "document_name": document_name,
                        "chunk_number": i+1,
                        "total_chunks": len(chunks),
                        "processing_timestamp": datetime.now().isoformat()
                    }
                )
                
                successful_chunks += 1
                
                # Add delay between chunks to be extra safe
                if i < len(chunks) - 1:  # Don't sleep after last chunk
                    await asyncio.sleep(5)  # 5 second pause between chunks
                    
            except Exception as e:
                print(f"❌ Failed chunk {i+1}: {str(e)}")
                failed_chunks.append((i+1, str(e)))
                
                # For rate limit errors, wait longer before continuing
                if "rate limit" in str(e).lower():
                    print("⏸️ Extended wait due to rate limit...")
                    await asyncio.sleep(120)  # 2 minute wait
        
        return {
            "total_chunks": len(chunks),
            "successful_chunks": successful_chunks,
            "failed_chunks": failed_chunks,
            "success_rate": successful_chunks / len(chunks) * 100
        }
```

## Advanced Strategies

### Strategy 1: Episode Consolidation

Instead of multiple small episodes, create fewer comprehensive episodes:

```python
async def consolidate_ingestion_approach(pdf_content, document_name):
    """
    Single episode approach for better rate limiting
    """
    # Process entire document as one episode (if under token limit)
    if len(pdf_content) < 50000:  # Adjust based on your LLM context window
        await graphiti.add_episode(
            name=f"{document_name}_complete",
            episode_body=pdf_content,
            source_description=f"Complete {document_name} document",
            metadata={
                "document_type": "medical_research",
                "ingestion_method": "single_episode",
                "processing_timestamp": datetime.now().isoformat()
            }
        )
    else:
        # Use the semantic chunking approach above
        await rate_limited_ingestion.ingest_large_pdf(pdf_content, document_name)
```

### Strategy 2: Provider-Specific Optimization

```python
# For OpenAI API (adjust based on your provider)
RATE_LIMITS = {
    "openai_gpt4": {
        "rpm": 500,      # requests per minute
        "tpm": 30000,    # tokens per minute
        "chunk_delay": 2  # seconds between chunks
    },
    "openai_gpt35": {
        "rpm": 3500,
        "tpm": 90000,
        "chunk_delay": 1
    },
    "azure_openai": {
        "rpm": 300,      # Often more restrictive
        "tpm": 40000,
        "chunk_delay": 3
    }
}

def configure_for_provider(provider_name):
    limits = RATE_LIMITS.get(provider_name, RATE_LIMITS["openai_gpt4"])
    
    # Set very conservative concurrency
    os.environ['SEMAPHORE_LIMIT'] = '1'
    
    return RateLimitedIngestion(
        requests_per_minute=limits["rpm"] // 10  # Use 10% of limit for safety
    )
```

### Strategy 3: Preprocessing Optimization

```python
def preprocess_pdf_for_graphiti(pdf_content):
    """
    Optimize content before ingestion to reduce API calls
    """
    # Remove redundant whitespace and formatting
    cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', pdf_content)
    
    # Extract and preserve key structured data
    structured_elements = {
        "tables": extract_tables(cleaned_content),
        "lists": extract_numbered_lists(cleaned_content),
        "references": extract_references(cleaned_content)
    }
    
    # Create context-rich chunks that minimize cross-chunk entity resolution
    optimized_chunks = create_self_contained_chunks(
        cleaned_content, 
        structured_elements,
        max_size=10000
    )
    
    return optimized_chunks
```

## Production Implementation

```python
import asyncio
import os
from datetime import datetime

class ProductionGraphitiIngestion:
    def __init__(self, provider="openai_gpt4"):
        # Set environment variables for rate limiting
        os.environ['SEMAPHORE_LIMIT'] = '1'
        os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'
        
        self.graphiti = Graphiti(
            graph_driver=your_driver,
            llm_client=your_llm_client
        )
        
        self.rate_limiter = RateLimitedIngestion(
            requests_per_minute=30  # Very conservative
        )
    
    async def ingest_large_document(self, pdf_path, document_name):
        """
        Production-ready large document ingestion
        """
        print(f"🚀 Starting ingestion of {document_name}")
        
        # Read and preprocess
        with open(pdf_path, 'r') as f:
            content = f.read()
        
        preprocessed_chunks = preprocess_pdf_for_graphiti(content)
        
        print(f"📊 Created {len(preprocessed_chunks)} optimized chunks")
        
        # Process with rate limiting
        result = await self.rate_limiter.ingest_large_pdf(
            "\n\n".join(preprocessed_chunks), 
            document_name
        )
        
        print(f"✅ Ingestion complete: {result['success_rate']:.1f}% success")
        return result

# Usage
async def main():
    ingestion_manager = ProductionGraphitiIngestion()
    
    result = await ingestion_manager.ingest_large_document(
        "LOG_CHESTXRAY.pdf",
        "NIH_ChestXray_Dataset"
    )
    
    print(f"Final result: {result}")

# Run the ingestion
asyncio.run(main())
```

## Key Recommendations

### Immediate Actions:
1. **Set `SEMAPHORE_LIMIT=1`** to force sequential processing
2. **Use semantic chunking** instead of arbitrary length chunking  
3. **Add delays** between chunk processing (5-10 seconds)
4. **Monitor your API usage** in real-time

### Long-term Optimizations:
1. **Upgrade to higher rate limits** if using production API tiers
2. **Consider local LLM models** for development/testing (Ollama + Graphiti)
3. **Implement retry logic** with exponential backoff
4. **Use batch processing** during off-peak hours

Chunking articles into multiple Episodes improved our results compared to treating each article as a single Episode. This approach generated more detailed knowledge, but requires careful rate limit management for large documents.

The key is balancing **semantic coherence** (fewer, larger chunks) with **API rate limits** (sequential processing with delays).