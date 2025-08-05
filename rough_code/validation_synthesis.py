from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI
from workflow_state import WorkflowState, ValidationResult, CitationSource
from observability import observability
from logging_config import get_logger

logger = get_logger("validation_synthesis")


class ValidatorAgent:
    """
    Validator Agent - Validates consistency between vector and graph retrieval results
    Reads: state.vector_docs?, state.graph_triples?
    Validates consistency via LLM or rules
    Writes: state.validation_passed, state.validation_errors, state.latency_ms["val"]
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.validation_prompt = """
        You are a data validation expert. Analyze the following retrieved information for consistency and accuracy.
        
        Query: {query}
        
        Vector Search Results:
        {vector_results}
        
        Graph Search Results:
        {graph_results}
        
        Validate the following:
        1. Do the results contradict each other?
        2. Are there any logical inconsistencies?
        3. Do the results actually address the query?
        4. What is the confidence level of the information?
        5. Are there any potential gaps or missing information?
        
        Provide your analysis in this format:
        VALIDATION: PASSED/FAILED
        CONFIDENCE: [0.0-1.0]
        CONSISTENCY_SCORE: [0.0-1.0]
        ERRORS: [list any critical errors or contradictions]
        WARNINGS: [list any concerns or potential issues]
        REASONING: [detailed explanation of your assessment]
        """
    
    def validate_results(self, state: WorkflowState) -> WorkflowState:
        """
        Validate consistency between different retrieval results
        """
        
        with observability.measure_agent_performance("val", state):
            try:
                query = state["query"]
                vector_docs = state.get("vector_docs", [])
                graph_triples = state.get("graph_triples", [])
                
                # For Happy Path, assume validation always passes
                # This simplifies the workflow to focus on the core functionality
                validation_result = ValidationResult(
                    passed=True,
                    errors=[],
                    confidence_score=0.8,  # Default good confidence
                    consistency_issues=[]
                )
                
                # If we have data, perform basic validation
                if vector_docs or graph_triples:
                    # Prepare data for validation
                    vector_summary = self._summarize_vector_results(vector_docs)
                    graph_summary = self._summarize_graph_results(graph_triples)
                    
                    # Get LLM validation
                    validation_response = self.llm.invoke(
                        self.validation_prompt.format(
                            query=query,
                            vector_results=vector_summary,
                            graph_results=graph_summary
                        )
                    )
                    
                    # Parse validation response
                    validation_result = self._parse_validation_response(validation_response.content)
                    
                    # Apply rule-based validation as backup
                    rule_based_validation = self._rule_based_validation(vector_docs, graph_triples, query)
                    
                    # Combine validations
                    final_validation = self._combine_validations(validation_result, rule_based_validation)
                else:
                    # No data but still pass for Happy Path
                    final_validation = ValidationResult(
                        passed=True,
                        errors=["No data retrieved but proceeding with Happy Path"],
                        confidence_score=0.5,
                        consistency_issues=[]
                    )
                
                # Update state
                state["validation_passed"] = final_validation.passed
                state["validation_errors"] = final_validation.errors
                state["validation_result"] = final_validation
                
                # Log validation results
                observability.log_validation_result(state)
                
                logger.info(
                    "validation_completed",
                    passed=final_validation.passed,
                    confidence=final_validation.confidence_score,
                    errors_count=len(final_validation.errors),
                    trace_id=state.get('trace_id')
                )
                
                return state
                
            except Exception as e:
                logger.error("validation_error", error=str(e), trace_id=state.get('trace_id'))
                # Fail safely - allow processing to continue for Happy Path
                validation_result = ValidationResult(
                    passed=True,
                    errors=[f"Validation error: {str(e)}"],
                    confidence_score=0.5,
                    consistency_issues=["Could not perform full validation"]
                )
                state["validation_passed"] = True
                state["validation_errors"] = validation_result.errors
                state["validation_result"] = validation_result
                return state
    
    def _summarize_vector_results(self, vector_docs: List[Dict[str, Any]]) -> str:
        """Create a summary of vector search results for validation"""
        if not vector_docs:
            return "No vector search results"
        
        summary_parts = []
        limit = 5  # Standard limit
        
        for i, doc in enumerate(vector_docs[:limit]):
            content_preview = doc.get("content", "")[:200]
            score = doc.get("score", 0)
            summary_parts.append(f"{i+1}. (Score: {score:.3f}) {content_preview}...")
        
        return "\n".join(summary_parts)
    
    def _summarize_graph_results(self, graph_triples: List[Dict[str, Any]]) -> str:
        """Create a summary of graph search results for validation"""
        if not graph_triples:
            return "No graph search results"
        
        summary_parts = []
        limit = 10  # Standard limit
        
        for i, triple in enumerate(graph_triples[:limit]):
            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "")
            obj = triple.get("object", "")
            is_retry_result = triple.get("is_retry_result", False)
            retry_marker = " (RETRY)" if is_retry_result else ""
            summary_parts.append(f"{i+1}. {subject} -> {predicate} -> {obj}{retry_marker}")
        
        return "\n".join(summary_parts)
    
    def _get_retry_validation_prompt(self) -> str:
        """
        Enhanced validation prompt for retry attempts
        More lenient criteria considering this is a retry
        """
        return """
        You are a data validation expert. This is a RETRY validation attempt.
        
        Query: {query}
        Retry Count: {retry_count}
        
        Vector Search Results:
        {vector_results}
        
        Graph Search Results:
        {graph_results}
        
        Since this is a retry attempt, be more lenient in your validation:
        1. Accept partial matches and related information
        2. Consider if the combined results provide useful context
        3. Focus on whether the results contribute to answering the query
        4. Lower the bar for consistency requirements
        5. Value breadth of information over perfect accuracy
        
        Provide your analysis in this format:
        VALIDATION: PASSED/FAILED
        CONFIDENCE: [0.0-1.0]
        CONSISTENCY_SCORE: [0.0-1.0]
        ERRORS: [list only critical errors]
        WARNINGS: [list any concerns]
        REASONING: [explain why this retry should pass/fail]
        """
    
    def _get_final_validation_prompt(self) -> str:
        """
        Final validation prompt when max retries reached
        Very lenient - proceed with whatever we have
        """
        return """
        You are a data validation expert. This is the FINAL validation attempt.
        
        Query: {query}
        
        Vector Search Results:
        {vector_results}
        
        Graph Search Results:
        {graph_results}
        
        This is the final validation after retries. Focus on:
        1. Is there ANY useful information to answer the query?
        2. Can we provide a reasonable response with caveats?
        3. Are there any completely false or harmful results?
        
        Be very lenient - we should proceed unless the results are harmful.
        
        Provide your analysis in this format:
        VALIDATION: PASSED/FAILED
        CONFIDENCE: [0.0-1.0]
        CONSISTENCY_SCORE: [0.0-1.0]
        ERRORS: [only list harmful or completely false information]
        WARNINGS: [quality issues to note in response]
        REASONING: [brief explanation]
        """
    
    def _combine_validations_with_retry_logic(self, llm_validation: ValidationResult, 
                                            rule_validation: ValidationResult, 
                                            retry_count: int, is_final: bool) -> ValidationResult:
        """
        Combine validation results considering retry context
        More lenient for retries, very lenient for final validation
        """
        if is_final:
            # Final validation - pass unless there are critical errors
            passed = not any("harmful" in error.lower() or "false" in error.lower() 
                           for error in llm_validation.errors + rule_validation.errors)
            confidence = max(0.3, min(llm_validation.confidence_score, rule_validation.confidence_score))
        elif retry_count > 0:
            # Retry validation - lower threshold
            passed = (llm_validation.confidence_score > 0.4 or rule_validation.confidence_score > 0.4)
            confidence = max(llm_validation.confidence_score, rule_validation.confidence_score)
        else:
            # Initial validation - standard threshold
            passed = llm_validation.passed and rule_validation.passed
            confidence = min(llm_validation.confidence_score, rule_validation.confidence_score)
        
        return ValidationResult(
            passed=passed,
            errors=llm_validation.errors + rule_validation.errors,
            confidence_score=confidence,
            consistency_issues=llm_validation.consistency_issues + rule_validation.consistency_issues,
            cross_validation_score=(llm_validation.confidence_score + rule_validation.confidence_score) / 2
        )
    
    def _parse_validation_response(self, response: str) -> ValidationResult:
        """Parse LLM validation response"""
        try:
            lines = response.split('\n')
            passed = False
            confidence_score = 0.5
            consistency_score = 0.5
            errors = []
            warnings = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('VALIDATION:'):
                    passed = 'PASSED' in line.upper()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence_score = float(line.split(':')[1].strip())
                    except:
                        confidence_score = 0.5
                elif line.startswith('CONSISTENCY_SCORE:'):
                    try:
                        consistency_score = float(line.split(':')[1].strip())
                    except:
                        consistency_score = 0.5
                elif line.startswith('ERRORS:'):
                    error_text = line.split('ERRORS:')[1].strip()
                    if error_text and error_text != '[]':
                        errors = [e.strip() for e in error_text.strip('[]').split(',')]
                elif line.startswith('WARNINGS:'):
                    warning_text = line.split('WARNINGS:')[1].strip()
                    if warning_text and warning_text != '[]':
                        warnings = [w.strip() for w in warning_text.strip('[]').split(',')]
            
            return ValidationResult(
                passed=passed,
                errors=errors,
                confidence_score=confidence_score,
                consistency_issues=warnings,
                cross_validation_score=consistency_score
            )
            
        except Exception as e:
            logger.warning("validation_parsing_failed", error=str(e))
            return ValidationResult(
                passed=True,  # Default to passing to avoid blocking
                errors=[f"Could not parse validation response: {str(e)}"],
                confidence_score=0.5
            )
    
    def _rule_based_validation(self, vector_docs: List[Dict[str, Any]], 
                             graph_triples: List[Dict[str, Any]], query: str, 
                             retry_count: int = 0) -> ValidationResult:
        """
        Apply rule-based validation as backup
        Enhanced version considers retry context for more lenient validation
        """
        errors = []
        warnings = []
        confidence = 1.0
        
        # Adjust thresholds based on retry count
        min_score_threshold = 0.3 if retry_count == 0 else 0.2  # Lower bar for retries
        confidence_penalty = 0.8 if retry_count == 0 else 0.9   # Less penalty for retries
        
        # Check if we have any results
        if not vector_docs and not graph_triples:
            errors.append("No results from any retrieval method")
            confidence = 0.0
        
        # Check vector results quality with retry-adjusted thresholds
        if vector_docs:
            avg_score = sum(doc.get("score", 0) for doc in vector_docs) / len(vector_docs)
            if avg_score < min_score_threshold:
                if retry_count > 0:
                    warnings.append(f"Lower quality scores in vector results (retry attempt {retry_count})")
                    confidence *= 0.9  # Less severe penalty for retries
                else:
                    warnings.append("Low average similarity scores in vector results")
                    confidence *= confidence_penalty
        
        # Check for contradictions with retry-sensitive logic
        if vector_docs and graph_triples:
            vector_text = " ".join([doc.get("content", "") for doc in vector_docs])
            graph_text = " ".join([f"{t.get('subject', '')} {t.get('predicate', '')} {t.get('object', '')}" 
                                 for t in graph_triples])
            
            # Simple contradiction detection (more lenient for retries)
            contradiction_keywords = ["not", "never", "opposite", "contrary", "false"]
            contradictions_found = sum(1 for keyword in contradiction_keywords 
                                     if keyword in vector_text.lower() and keyword in graph_text.lower())
            
            if contradictions_found > 0:
                if retry_count > 0:
                    warnings.append(f"Some inconsistencies noted between sources (retry {retry_count})")
                    confidence *= 0.85  # Less severe for retries
                else:
                    warnings.append("Potential contradictions detected between sources")
                    confidence *= 0.7
        
        # For retry attempts, be more forgiving about data quality
        if retry_count > 0 and confidence < 0.4:
            confidence = max(confidence, 0.4)  # Minimum threshold for retries
            warnings.append("Quality adjusted for retry attempt")
        
        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            confidence_score=confidence,
            consistency_issues=warnings
        )
    
    def _combine_validations(self, llm_validation: ValidationResult, 
                           rule_validation: ValidationResult) -> ValidationResult:
        """Combine LLM and rule-based validation results"""
        # Use the more conservative approach
        passed = llm_validation.passed and rule_validation.passed
        
        # Combine errors and warnings
        all_errors = llm_validation.errors + rule_validation.errors
        all_warnings = llm_validation.consistency_issues + rule_validation.consistency_issues
        
        # Average confidence scores
        avg_confidence = (llm_validation.confidence_score + rule_validation.confidence_score) / 2
        
        return ValidationResult(
            passed=passed,
            errors=list(set(all_errors)),  # Remove duplicates
            confidence_score=avg_confidence,
            consistency_issues=list(set(all_warnings)),
            cross_validation_score=llm_validation.cross_validation_score
        )


class AnswerSynthesisAgent:
    """
    Answer Synthesis Agent - Composes final answer with citations
    Reads: state.vector_docs, state.graph_triples, state.validation_passed
    LLM composition with citations
    Writes: state.answer, state.citations, state.latency_ms["ans"], state.memory_usage["ans"]
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.synthesis_prompt = """
        You are an expert AI assistant that synthesizes information from multiple sources to provide comprehensive answers.
        
        Query: {query}
        
        Validation Status: {validation_status}
        Confidence Score: {confidence_score}
        
        Vector Search Results:
        {vector_results}
        
        Graph Search Results:
        {graph_results}
        
        Validation Issues (if any):
        {validation_issues}
        
        Instructions:
        1. Synthesize a comprehensive answer using information from both sources
        2. Clearly cite sources using [Vector-1], [Graph-1] format
        3. If validation failed, mention limitations and uncertainties
        4. If information is contradictory, present both perspectives
        5. Provide a confidence assessment for your answer
        6. Keep the answer well-structured and readable
        
        Format your response as:
        ANSWER: [Your comprehensive answer with citations]
        CONFIDENCE: [0.0-1.0]
        SOURCES_USED: [Vector: X documents, Graph: Y triples]
        LIMITATIONS: [Any limitations or uncertainties]
        """
    
    def synthesize_answer(self, state: WorkflowState) -> WorkflowState:
        """
        Synthesize final answer from all available information
        
        Enhanced version with retry context:
        - Considers retry attempts and validation failures
        - Adjusts confidence and response based on retry status
        - Provides appropriate caveats for retry scenarios
        """
        
        with observability.measure_agent_performance("ans", state):
            try:
                query = state["query"]
                vector_docs = state.get("vector_docs", [])
                graph_triples = state.get("graph_triples", [])
                validation_result = state.get("validation_result")
                validation_passed = state.get("validation_passed", True)
                retry_count = state.get("retry_count", 0)
                is_final_validation = state.get("is_final_validation", False)
                
                # Handle case where no data is available
                if not vector_docs and not graph_triples:
                    state["answer"] = "I don't have enough information to answer this query based on the available data sources."
                    state["citations"] = []
                    state["confidence_score"] = 0.0
                    state["status"] = "completed"
                    return state
                
                # Prepare data for synthesis with retry context
                vector_content = self._format_vector_content_with_retry(vector_docs, retry_count)
                graph_content = self._format_graph_content_with_retry(graph_triples, retry_count)
                validation_info = self._format_validation_info(validation_result)
                
                # Use appropriate synthesis prompt based on retry status
                if retry_count > 0:
                    synthesis_prompt = self._get_retry_synthesis_prompt()
                else:
                    synthesis_prompt = self.synthesis_prompt
                
                # Generate synthesis with retry context
                synthesis_response = self.llm.invoke(
                    synthesis_prompt.format(
                        query=query,
                        validation_status="PASSED" if validation_passed else "FAILED",
                        confidence_score=validation_result.confidence_score if validation_result else 0.5,
                        vector_results=vector_content,
                        graph_results=graph_content,
                        validation_issues=validation_info,
                        retry_count=retry_count,
                        is_final_validation=is_final_validation
                    )
                )
                
                # Parse synthesis response with retry context
                answer, confidence, citations = self._parse_synthesis_response_with_retry(
                    synthesis_response.content, vector_docs, graph_triples, retry_count
                )
                
                # Adjust status based on retry and validation
                if retry_count > 0:
                    status = "completed_with_retry" if validation_passed else "completed_with_retry_and_issues"
                else:
                    status = "completed" if validation_passed else "completed_with_issues"
                
                # Update state
                state["answer"] = answer
                state["citations"] = citations
                state["confidence_score"] = confidence
                state["status"] = status
                
                logger.info(
                    "answer_synthesis_completed",
                    answer_length=len(answer),
                    citations_count=len(citations),
                    confidence=confidence,
                    retry_count=retry_count,
                    validation_passed=validation_passed,
                    trace_id=state.get('trace_id')
                )
                
                return state
                
            except Exception as e:
                logger.error("synthesis_error", error=str(e), retry_count=retry_count, trace_id=state.get('trace_id'))
                state["answer"] = f"I encountered an error while synthesizing the answer: {str(e)}"
                state["citations"] = []
                state["confidence_score"] = 0.0
                state["status"] = "failed"
                state["errors"] = state.get("errors", []) + [f"Synthesis error: {str(e)}"]
                return state
    
    def _format_vector_content_with_retry(self, vector_docs: List[Dict[str, Any]], retry_count: int) -> str:
        """
        Format vector search results for synthesis with retry context
        Marks retry results and provides appropriate context
        """
        if not vector_docs:
            return "No vector search results available."
        
        formatted_docs = []
        for i, doc in enumerate(vector_docs):
            content = doc.get("content", "")[:500]  # Limit content length
            score = doc.get("score", 0)
            is_retry_result = doc.get("is_retry_result", False)
            retry_marker = f" [RETRY-{retry_count}]" if is_retry_result else ""
            formatted_docs.append(f"[Vector-{i+1}]{retry_marker} (Relevance: {score:.3f}) {content}")
        
        return "\n\n".join(formatted_docs)
    
    def _format_graph_content_with_retry(self, graph_triples: List[Dict[str, Any]], retry_count: int) -> str:
        """
        Format graph search results for synthesis with retry context
        Marks retry results and provides appropriate context
        """
        if not graph_triples:
            return "No graph search results available."
        
        formatted_triples = []
        for i, triple in enumerate(graph_triples):
            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "")
            obj = triple.get("object", "")
            is_retry_result = triple.get("is_retry_result", False)
            retry_marker = f" [RETRY-{retry_count}]" if is_retry_result else ""
            formatted_triples.append(f"[Graph-{i+1}]{retry_marker} {subject} -> {predicate} -> {obj}")
        
        return "\n".join(formatted_triples)
    
    def _get_retry_synthesis_prompt(self) -> str:
        """
        Enhanced synthesis prompt for retry scenarios
        More forgiving and includes retry context
        """
        return """
        You are an expert AI assistant that synthesizes information from multiple sources to provide comprehensive answers.
        
        RETRY CONTEXT: This is attempt {retry_count} at answering this query after validation issues.
        
        Query: {query}
        
        Validation Status: {validation_status}
        Confidence Score: {confidence_score}
        Is Final Validation: {is_final_validation}
        
        Vector Search Results:
        {vector_results}
        
        Graph Search Results:
        {graph_results}
        
        Validation Issues (if any):
        {validation_issues}
        
        Instructions for RETRY synthesis:
        1. Be more lenient with information quality since this is a retry
        2. Synthesize the best possible answer from available information
        3. Clearly cite sources using [Vector-1], [Graph-1] format
        4. Include appropriate caveats about retry context and limitations
        5. If validation failed, be transparent about quality concerns
        6. Provide the best answer possible while being honest about limitations
        7. For retry results marked [RETRY-X], note they came from enhanced search
        
        Format your response as:
        ANSWER: [Your comprehensive answer with citations and retry context]
        CONFIDENCE: [0.0-1.0] (adjusted for retry context)
        SOURCES_USED: [Vector: X documents, Graph: Y triples]
        LIMITATIONS: [Retry context, validation issues, quality concerns]
        RETRY_NOTES: [Specific notes about retry process and results]
        """
    
    def _parse_synthesis_response_with_retry(self, response: str, vector_docs: List[Dict[str, Any]], 
                                           graph_triples: List[Dict[str, Any]], retry_count: int) -> tuple:
        """
        Parse synthesis response with retry context
        Adjusts confidence and adds retry information to citations
        """
        answer = ""
        confidence = 0.5
        citations = []
        
        try:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('ANSWER:'):
                    answer = line[7:].strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line[11:].strip())
                        # Adjust confidence for retry attempts
                        if retry_count > 0:
                            confidence = max(confidence * 0.9, 0.3)  # Slight penalty for retries
                    except ValueError:
                        confidence = 0.5
            
            # Extract citations from answer with retry context
            citations = self._extract_citations_with_retry(answer, vector_docs, graph_triples, retry_count)
            
        except Exception as e:
            logger.error("synthesis_parsing_error", error=str(e), retry_count=retry_count)
            answer = "Unable to parse synthesis response properly."
            confidence = 0.3
            citations = []
        
        return answer, confidence, citations
    
    def _extract_citations_with_retry(self, answer: str, vector_docs: List[Dict[str, Any]], 
                                    graph_triples: List[Dict[str, Any]], retry_count: int) -> List[CitationSource]:
        """
        Extract citations from answer with retry context
        Marks citations that came from retry attempts
        """
        citations = []
        
        # Extract vector citations
        import re
        vector_citations = re.findall(r'\[Vector-(\d+)\]', answer)
        for citation_num in vector_citations:
            try:
                doc_index = int(citation_num) - 1
                if 0 <= doc_index < len(vector_docs):
                    doc = vector_docs[doc_index]
                    is_retry_result = doc.get("is_retry_result", False)
                    citation = CitationSource(
                        type="vector",
                        content=doc.get("content", "")[:200],
                        score=doc.get("score", 0.0),
                        source_id=doc.get("id"),
                        metadata={
                            **doc.get("metadata", {}),
                            "is_retry_result": is_retry_result,
                            "retry_count": retry_count if is_retry_result else 0
                        }
                    )
                    citations.append(citation)
            except (ValueError, IndexError):
                continue
        
        # Extract graph citations
        graph_citations = re.findall(r'\[Graph-(\d+)\]', answer)
        for citation_num in graph_citations:
            try:
                triple_index = int(citation_num) - 1
                if 0 <= triple_index < len(graph_triples):
                    triple = graph_triples[triple_index]
                    is_retry_result = triple.get("is_retry_result", False)
                    citation = CitationSource(
                        type="graph",
                        content=f"{triple.get('subject', '')} -> {triple.get('predicate', '')} -> {triple.get('object', '')}",
                        score=1.0,  # Graph triples don't have scores
                        source_id=triple.get("query"),
                        metadata={
                            **triple.get("metadata", {}),
                            "is_retry_result": is_retry_result,
                            "retry_count": retry_count if is_retry_result else 0
                        }
                    )
                    citations.append(citation)
            except (ValueError, IndexError):
                continue
        
        return citations
    
    def _format_validation_info(self, validation_result: ValidationResult) -> str:
        """Format validation information"""
        if not validation_result:
            return "Validation information not available."
        
        info_parts = []
        if validation_result.errors:
            info_parts.append(f"Errors: {', '.join(validation_result.errors)}")
        if validation_result.consistency_issues:
            info_parts.append(f"Warnings: {', '.join(validation_result.consistency_issues)}")
        
        return "\n".join(info_parts) if info_parts else "No validation issues detected."
    
    def _parse_synthesis_response(self, response: str, vector_docs: List[Dict[str, Any]], 
                                graph_triples: List[Dict[str, Any]]) -> tuple:
        """Parse the synthesis response and extract answer, confidence, and citations"""
        try:
            lines = response.split('\n')
            answer = ""
            confidence = 0.5
            
            for line in lines:
                line = line.strip()
                if line.startswith('ANSWER:'):
                    answer = line.split('ANSWER:')[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':')[1].strip())
                    except:
                        confidence = 0.5
            
            # Extract citations from the answer
            citations = self._extract_citations(answer, vector_docs, graph_triples)
            
            return answer, confidence, citations
            
        except Exception as e:
            logger.warning("synthesis_parsing_failed", error=str(e))
            return response, 0.5, []
    
    def _extract_citations(self, answer: str, vector_docs: List[Dict[str, Any]], 
                         graph_triples: List[Dict[str, Any]]) -> List[CitationSource]:
        """Extract citation information from the answer"""
        citations = []
        
        # Find vector citations
        import re
        vector_matches = re.findall(r'\[Vector-(\d+)\]', answer)
        for match in vector_matches:
            idx = int(match) - 1
            if 0 <= idx < len(vector_docs):
                doc = vector_docs[idx]
                citation = CitationSource(
                    type="vector",
                    content=doc.get("content", "")[:200],
                    score=doc.get("score", 0),
                    metadata=doc.get("metadata", {}),
                    source_id=f"vector-{idx+1}"
                )
                citations.append(citation)
        
        # Find graph citations
        graph_matches = re.findall(r'\[Graph-(\d+)\]', answer)
        for match in graph_matches:
            idx = int(match) - 1
            if 0 <= idx < len(graph_triples):
                triple = graph_triples[idx]
                citation = CitationSource(
                    type="graph",
                    content=f"{triple.get('subject', '')} -> {triple.get('predicate', '')} -> {triple.get('object', '')}",
                    score=1.0,  # Graph triples don't have similarity scores
                    metadata=triple.get("metadata", {}),
                    source_id=f"graph-{idx+1}"
                )
                citations.append(citation)
        
        return citations
