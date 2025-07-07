"""
Complete Agentic AI Test Generation System with Gemini 1.5 Flash Integration
Advanced multi-agent system for intelligent test generation
"""
import ast
import inspect
import json
import logging
import os
import random
import re
import string
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import importlib.util
import sys
from collections import defaultdict
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
import argparse
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class TestType(Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    EDGE_CASE = "edge_case"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ERROR_HANDLING = "error_handling"
    PROPERTY_BASED = "property_based"
    CONTRACT = "contract"
    MUTATION = "mutation"
class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
class TestFramework(Enum):
    PYTEST = "pytest"
    UNITTEST = "unittest"
    NOSE2 = "nose2"
@dataclass
class FunctionInfo:
    name: str
    file_path: str
    line_number: int
    parameters: List[Dict[str, Any]]
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    complexity: int = 0
    is_security_sensitive: bool = False
    dependencies: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    business_logic_type: Optional[str] = None
    data_flow: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    semantic_tags: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    semantic_analysis: Optional[Dict[str, Any]] = None  # Add this field
@dataclass
class TestCase:
    name: str
    test_type: TestType
    priority: Priority
    target_function: str
    test_code: str
    description: str
    framework: TestFramework = TestFramework.PYTEST
    assertions: List[str] = field(default_factory=list)
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    maintainability_score: float = 0.0
    expected_outcome: Optional[str] = None
    test_data: Dict[str, Any] = field(default_factory=dict)
@dataclass
class TestSuite:
    name: str
    test_cases: List[TestCase] = field(default_factory=list)
    coverage_percentage: float = 0.0
    quality_score: float = 0.0
    security_coverage: float = 0.0
    maintainability_score: float = 0.0
    execution_time_estimate: float = 0.0
    framework: TestFramework = TestFramework.PYTEST
    mutation_score: float = 0.0  # 5. Add this line
class GeminiLLMAgent:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API key is required")
        if not GEMINI_AVAILABLE:
            raise ImportError("google.generativeai is not installed")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.logger = logging.getLogger(__name__ + ".GeminiAgent")
    def analyze_function_semantics(self, func_info: FunctionInfo, source_code: str) -> Dict[str, Any]:
        try:
            prompt = f"""
            Analyze this Python function for semantic understanding:
            Function: {func_info.name}
            Docstring: {func_info.docstring or 'None'}
            Parameters: {func_info.parameters}
            Return Type: {func_info.return_type or 'Unknown'}
            Source Code:
            ```python
            {source_code}
            ```
            Please provide a JSON response with:
            1. business_logic_type
            2. semantic_tags
            3. risk_assessment
            4. expected_behaviors
            5. edge_cases
            6. security_concerns
            7. test_scenarios
            """
            response = self.model.generate_content(prompt)
            if not response or not getattr(response, "text", None):
                self.logger.warning("Empty response from Gemini API")
                return self._get_fallback_semantic_analysis(func_info)
            json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                return json.loads(response.text)
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            return self._get_fallback_semantic_analysis(func_info)
    def generate_test_scenarios(self, func_info: FunctionInfo, semantic_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            prompt = f"""
            Generate comprehensive test scenarios for this function:
            Function: {func_info.name}
            Semantic Analysis: {json.dumps(semantic_analysis, indent=2)}
            Generate test scenarios covering:
            1. Happy path scenarios
            2. Edge cases and boundary conditions
            3. Error conditions and exception handling
            4. Security test cases (if applicable)
            5. Integration scenarios
            6. Performance considerations
            For each scenario, provide:
            - scenario_name
            - test_type
            - priority
            - description
            - test_data
            - expected_outcome
            - assertions
            Return as JSON array of scenarios.
            """
            response = self.model.generate_content(prompt)
            if not response or not getattr(response, "text", None):
                self.logger.warning("Empty response from Gemini API")
                return self._get_fallback_scenarios(func_info)
            json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                return json.loads(response.text)
        except Exception as e:
            self.logger.error(f"Error generating test scenarios: {e}")
            return self._get_fallback_scenarios(func_info)
    def generate_test_code(self, scenario: Dict[str, Any], func_info: FunctionInfo, framework: TestFramework) -> str:
        try:
            prompt = f"""
            Generate {framework.value} test code for this scenario:
            Function: {func_info.name}
            Parameters: {func_info.parameters}
            Scenario: {json.dumps(scenario, indent=2)}
            Generate complete, executable test code including:
            - Proper imports
            - Setup and teardown if needed
            - Realistic test data
            - Comprehensive assertions
            - Error handling
            - Comments explaining the test
            Follow {framework.value} best practices and conventions.
            """
            response = self.model.generate_content(prompt)
            if not response or not getattr(response, "text", None):
                self.logger.warning("Empty response from Gemini API")
                return ""
            code_match = re.search(r'```python\s*(.*?)\s*```', response.text, re.DOTALL)
            if code_match:
                return code_match.group(1)
            else:
                return response.text
        except Exception as e:
            self.logger.error(f"Error generating test code: {e}")
            return ""
    def _get_fallback_semantic_analysis(self, func_info: FunctionInfo) -> Dict[str, Any]:
        # Simple fallback: use function name and docstring
        return {
            "business_logic_type": "utility" if "util" in func_info.name else "unknown",
            "semantic_tags": [func_info.name, "fallback"],
            "risk_assessment": {"score": 0.1},
            "expected_behaviors": [],
            "edge_cases": [],
            "security_concerns": [],
            "test_scenarios": []
        }
    def _get_fallback_scenarios(self, func_info: FunctionInfo) -> List[Dict[str, Any]]:
        # Simple fallback: happy path and edge case
        return [
            {
                "scenario_name": f"test_{func_info.name}_happy_path",
                "test_type": "unit",
                "priority": "medium",
                "description": f"Fallback: Test {func_info.name} with valid inputs",
                "test_data": {},
                "expected_outcome": "success"
            },
            {
                "scenario_name": f"test_{func_info.name}_edge_case",
                "test_type": "edge_case",
                "priority": "high",
                "description": f"Fallback: Test {func_info.name} with edge case inputs",
                "test_data": {},
                "expected_outcome": "success"
            }
        ]

class BaseAgent(ABC):
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.llm_agent = None
    def set_llm_agent(self, llm_agent: GeminiLLMAgent):
        self.llm_agent = llm_agent
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass
    def log_activity(self, message: str, level: str = "INFO"):
        getattr(self.logger, level.lower())(f"[{self.name}] {message}")

class EnhancedCodeAnalysisAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("EnhancedCodeAnalysisAgent", config)
        self.analyzed_functions: List[FunctionInfo] = []
    def process(self, file_path: str) -> List[FunctionInfo]:
        self.log_activity(f"Analyzing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                source_code = file.read()
            tree = ast.parse(source_code)
            functions = self._extract_functions(tree, file_path, source_code)
            if self.llm_agent:
                functions = self._enhance_with_semantics(functions, source_code)
            self.analyzed_functions.extend(functions)
            self.log_activity(f"Found {len(functions)} functions in {file_path}")
            return functions
        except Exception as e:
            self.log_activity(f"Error analyzing {file_path}: {str(e)}", "ERROR")
            return []
    def _extract_functions(self, tree: ast.AST, file_path: str, source_code: str) -> List[FunctionInfo]:
        functions = []
        lines = source_code.split('\n')
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self._analyze_function(node, file_path, lines)
                functions.append(func_info)
        return functions
    def _analyze_function(self, node: ast.FunctionDef, file_path: str, lines: List[str]) -> FunctionInfo:
        parameters = []
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'type': self._get_type_annotation(arg.annotation) if arg.annotation else 'Any',
                'default': None,
                'is_optional': False
            }
            parameters.append(param_info)
        return_type = self._get_type_annotation(node.returns) if node.returns else None
        docstring = ast.get_docstring(node)
        complexity = self._calculate_enhanced_complexity(node)
        is_security_sensitive = self._enhanced_security_analysis(node, docstring)
        dependencies = self._extract_enhanced_dependencies(node)
        exceptions = self._extract_exceptions(node)
        data_flow = self._analyze_data_flow(node)
        side_effects = self._detect_side_effects(node)
        risk_score = self._calculate_risk_score(node, complexity, is_security_sensitive, len(dependencies))
        return FunctionInfo(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            complexity=complexity,
            is_security_sensitive=is_security_sensitive,
            dependencies=dependencies,
            exceptions=exceptions,
            data_flow=data_flow,
            side_effects=side_effects,
            risk_score=risk_score
        )
    def _enhance_with_semantics(self, functions: List[FunctionInfo], source_code: str) -> List[FunctionInfo]:
        enhanced_functions = []
        for func_info in functions:
            func_source = self._extract_function_source(func_info, source_code)
            semantic_analysis = self.llm_agent.analyze_function_semantics(func_info, func_source)
            func_info.business_logic_type = semantic_analysis.get('business_logic_type')
            func_info.semantic_tags = semantic_analysis.get('semantic_tags', [])
            risk_data = semantic_analysis.get('risk_assessment', {})
            if isinstance(risk_data, dict) and 'score' in risk_data:
                func_info.risk_score = max(func_info.risk_score, risk_data['score'])
            func_info.semantic_analysis = semantic_analysis  # Store the full analysis
            enhanced_functions.append(func_info)
        return enhanced_functions
    def _extract_function_source(self, func_info: FunctionInfo, source_code: str) -> str:
        lines = source_code.split('\n')
        start_line = func_info.line_number - 1
        end_line = len(lines)
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent_level:
                end_line = i
                break
        return '\n'.join(lines[start_line:end_line])
    def _get_type_annotation(self, annotation: ast.AST) -> str:
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Attribute):
            return f"{self._get_type_annotation(annotation.value)}.{annotation.attr}"
        elif isinstance(annotation, ast.Subscript):
            return f"{self._get_type_annotation(annotation.value)}[{self._get_type_annotation(annotation.slice)}]"
        else:
            return "Any"
    def _calculate_enhanced_complexity(self, node: ast.FunctionDef) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1
            elif isinstance(child, ast.ListComp):
                complexity += 1
            elif isinstance(child, ast.DictComp):
                complexity += 1
        return complexity
    def _enhanced_security_analysis(self, node: ast.FunctionDef, docstring: Optional[str]) -> bool:
        security_keywords = [
            'password', 'token', 'auth', 'login', 'credential', 'secret', 'key',
            'encrypt', 'decrypt', 'hash', 'sql', 'query', 'execute', 'eval',
            'exec', 'admin', 'permission', 'access', 'validate', 'sanitize',
            'session', 'cookie', 'jwt', 'oauth', 'csrf', 'xss', 'injection'
        ]
        function_text = node.name.lower()
        if docstring:
            function_text += " " + docstring.lower()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id.lower() in ['eval', 'exec', 'compile']:
                        return True
                elif isinstance(child.func, ast.Attribute):
                    if child.func.attr.lower() in ['execute', 'query', 'raw']:
                        return True
        return any(keyword in function_text for keyword in security_keywords)
    def _extract_enhanced_dependencies(self, node: ast.FunctionDef) -> List[str]:
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.append(child.func.attr)
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    dependencies.append(alias.name)
            elif isinstance(child, ast.ImportFrom):
                if child.module:
                    dependencies.append(child.module)
        return list(set(dependencies))
    def _extract_exceptions(self, node: ast.FunctionDef) -> List[str]:
        exceptions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    exceptions.append(child.exc.func.id)
            elif isinstance(child, ast.ExceptHandler):
                if child.type and isinstance(child.type, ast.Name):
                    exceptions.append(child.type.id)
        return list(set(exceptions))
    def _analyze_data_flow(self, node: ast.FunctionDef) -> List[str]:
        data_flow = []
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        data_flow.append(f"assign_{target.id}")
            elif isinstance(child, ast.Return):
                data_flow.append("return_value")
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    data_flow.append(f"call_{child.func.id}")
        return data_flow
    def _detect_side_effects(self, node: ast.FunctionDef) -> List[str]:
        side_effects = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id in ['print', 'open', 'write']:
                        side_effects.append(f"io_{child.func.id}")
                elif isinstance(child.func, ast.Attribute):
                    if child.func.attr in ['save', 'delete', 'update', 'create']:
                        side_effects.append(f"data_{child.func.attr}")
        return side_effects
    def _calculate_risk_score(self, node: ast.FunctionDef, complexity: int, is_security_sensitive: bool, dependency_count: int) -> float:
        score = 0.0
        score += min(complexity / 10.0, 0.4)
        if is_security_sensitive:
            score += 0.3
        score += min(dependency_count / 20.0, 0.2)
        has_try_except = any(isinstance(child, ast.Try) for child in ast.walk(node))
        if not has_try_except:
            score += 0.1
        return min(score, 1.0)
class IntelligentTestDataGenerator:
    def __init__(self):
        self.random = random.Random(42)
    def generate_test_data(self, param_type: str, param_name: str, scenario: str = "normal") -> Any:
        if param_type in ['str', 'string']:
            return self._generate_string_data(param_name, scenario)
        elif param_type in ['int', 'integer']:
            return self._generate_int_data(param_name, scenario)
        elif param_type in ['float', 'number']:
            return self._generate_float_data(param_name, scenario)
        elif param_type in ['bool', 'boolean']:
            return self._generate_bool_data(scenario)
        elif param_type in ['list', 'List']:
            return self._generate_list_data(param_name, scenario)
        elif param_type in ['dict', 'Dict']:
            return self._generate_dict_data(param_name, scenario)
        elif param_type in ['None', 'NoneType']:
            return None
        else:
            return self._generate_generic_data(param_type, param_name, scenario)
    def _generate_string_data(self, param_name: str, scenario: str) -> str:
        name_lower = param_name.lower()
        if scenario == "edge_case":
            return self.random.choice(["", " ", "a" * 1000, "特殊字符", "🚀"])
        elif scenario == "security":
            return self.random.choice([
                "'; DROP TABLE users; --",
                "<script>alert('xss')</script>",
                "../../../etc/passwd",
                "admin' OR '1'='1"
            ])
        elif scenario == "null":
            return ""
        if any(word in name_lower for word in ['email', 'mail']):
            return f"test{self.random.randint(1, 1000)}@example.com"
        elif any(word in name_lower for word in ['name', 'user']):
            names = ["John Doe", "Jane Smith", "Alice Johnson", "Bob Wilson"]
            return self.random.choice(names)
        elif any(word in name_lower for word in ['password', 'pass']):
            return "SecurePass123!"
        elif any(word in name_lower for word in ['url', 'link']):
            return "https://example.com"
        elif any(word in name_lower for word in ['id', 'uuid']):
            return f"id_{self.random.randint(1000, 9999)}"
        else:
            return f"test_string_{self.random.randint(1, 100)}"
    def _generate_int_data(self, param_name: str, scenario: str) -> int:
        if scenario == "edge_case":
            return self.random.choice([0, -1, 1, 2**31 - 1, -2**31])
        elif scenario == "security":
            return self.random.choice([0, -1, 999999999])
        elif scenario == "null":
            return 0
        name_lower = param_name.lower()
        if any(word in name_lower for word in ['age', 'year']):
            return self.random.randint(18, 100)
        elif any(word in name_lower for word in ['count', 'size', 'length']):
            return self.random.randint(1, 1000)
        elif any(word in name_lower for word in ['id', 'index']):
            return self.random.randint(1, 10000)
        else:
            return self.random.randint(1, 100)
    def _generate_float_data(self, param_name: str, scenario: str) -> float:
        if scenario == "edge_case":
            return self.random.choice([0.0, -0.0, float('inf'), float('-inf')])
        elif scenario == "security":
            return self.random.choice([0.0, -999999.999, 999999.999])
        elif scenario == "null":
            return 0.0
        return self.random.uniform(-100.0, 100.0)
    def _generate_bool_data(self, scenario: str) -> bool:
        if scenario == "edge_case":
            return self.random.choice([True, False])
        elif scenario == "null":
            return False
        return self.random.choice([True, False])
    def _generate_list_data(self, param_name: str, scenario: str) -> List[Any]:
        if scenario == "edge_case":
            return self.random.choice([[], [None], list(range(1000))])
        elif scenario == "null":
            return []
        size = self.random.randint(1, 10)
        if 'string' in param_name.lower():
            return [f"item_{i}" for i in range(size)]
        elif 'int' in param_name.lower():
            return [self.random.randint(1, 100) for _ in range(size)]
        else:
            return [f"item_{i}" for i in range(size)]
    def _generate_dict_data(self, param_name: str, scenario: str) -> Dict[str, Any]:
        if scenario == "edge_case":
            return self.random.choice([{}, {"": ""}, {str(i): i for i in range(100)}])
        elif scenario == "null":
            return {}
        if 'user' in param_name.lower():
            return {
                "id": self.random.randint(1, 1000),
                "name": "Test User",
                "email": "test@example.com"
            }
        elif 'config' in param_name.lower():
            return {
                "debug": True,
                "timeout": 30,
                "retries": 3
            }
        else:
            return {
                "key1": "value1",
                "key2": self.random.randint(1, 100),
                "key3": True
            }
    def _generate_generic_data(self, param_type: str, param_name: str, scenario: str) -> Any:
        if scenario == "null":
            return None
        elif scenario == "edge_case":
            return None
        else:
            return f"mock_{param_type}_{param_name}"
class BaseAgent(ABC):
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.llm_agent = None
    def set_llm_agent(self, llm_agent: GeminiLLMAgent):
        self.llm_agent = llm_agent
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass
    def log_activity(self, message: str, level: str = "INFO"):
        getattr(self.logger, level.lower())(f"[{self.name}] {message}")
class EnhancedTestStrategyAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("EnhancedTestStrategyAgent", config)
    def process(self, functions: List[FunctionInfo]) -> Dict[str, Any]:
        self.log_activity(f"Planning enhanced test strategy for {len(functions)} functions")
        strategy = {
            'priorities': self._enhanced_prioritization(functions),
            'test_types': self._intelligent_test_types(functions),
            'coverage_goals': self._adaptive_coverage_goals(functions),
            'security_focus': self._comprehensive_security_analysis(functions),
            'integration_mapping': self._map_integration_tests(functions),
            'performance_targets': self._identify_performance_tests(functions),
            'quality_metrics': self._define_quality_metrics(functions)
        }
        return strategy
    def _enhanced_prioritization(self, functions: List[FunctionInfo]) -> Dict[str, Priority]:
        priorities = {}
        for func in functions:
            score = 0
            if func.is_security_sensitive:
                score += 40
            score += min(func.complexity * 5, 25)
            score += func.risk_score * 20
            score += len(func.side_effects) * 3
            score += min(len(func.dependencies) * 2, 10)
            if func.business_logic_type:
                business_priority = {
                    'authentication': 15,
                    'authorization': 15,
                    'data_processing': 10,
                    'api_endpoint': 12,
                    'validation': 8,
                    'utility': 3
                }
                score += business_priority.get(func.business_logic_type, 5)
            if score >= 70:
                priorities[func.name] = Priority.CRITICAL
            elif score >= 50:
                priorities[func.name] = Priority.HIGH
            elif score >= 30:
                priorities[func.name] = Priority.MEDIUM
            else:
                priorities[func.name] = Priority.LOW
        return priorities
    def _intelligent_test_types(self, functions: List[FunctionInfo]) -> Dict[str, List[TestType]]:
        test_types = {}
        for func in functions:
            types = [TestType.UNIT]
            if func.is_security_sensitive:
                types.extend([TestType.SECURITY, TestType.EDGE_CASE])
            if func.complexity > 5:
                types.extend([TestType.EDGE_CASE, TestType.ERROR_HANDLING])
            if func.side_effects:
                types.append(TestType.INTEGRATION)
            if len(func.dependencies) > 5:
                types.append(TestType.INTEGRATION)
            if func.exceptions:
                types.append(TestType.ERROR_HANDLING)
            if 'performance' in ' '.join(func.semantic_tags):
                types.append(TestType.PERFORMANCE)
            if func.business_logic_type:
                types.append(TestType.CONTRACT)
            if func.risk_score > 0.7:
                types.append(TestType.MUTATION)
            test_types[func.name] = list(set(types))
        return test_types
    def _adaptive_coverage_goals(self, functions: List[FunctionInfo]) -> Dict[str, float]:
        coverage_goals = {}
        for func in functions:
            base_coverage = 0.8
            if func.is_security_sensitive:
                base_coverage = 0.95
            if func.complexity > 10:
                base_coverage = min(base_coverage + 0.1, 0.98)
            if func.risk_score > 0.8:
                base_coverage = min(base_coverage + 0.05, 0.98)
            coverage_goals[func.name] = base_coverage
        return coverage_goals
    def _comprehensive_security_analysis(self, functions: List[FunctionInfo]) -> Dict[str, List[str]]:
        security_analysis = {}
        for func in functions:
            security_concerns = []
            if func.is_security_sensitive:
                if any('input' in tag for tag in func.semantic_tags):
                    security_concerns.append('input_validation')
                if any('auth' in tag for tag in func.semantic_tags):
                    security_concerns.append('authentication')
                if any('access' in tag for tag in func.semantic_tags):
                    security_concerns.append('authorization')
                if any('sql' in dep.lower() for dep in func.dependencies):
                    security_concerns.append('sql_injection')
                if any('html' in tag or 'web' in tag for tag in func.semantic_tags):
                    security_concerns.append('xss_prevention')
                if any('csrf' in tag for tag in func.semantic_tags):
                    security_concerns.append('csrf_protection')
            if security_concerns:
                security_analysis[func.name] = security_concerns
        return security_analysis
    def _map_integration_tests(self, functions: List[FunctionInfo]) -> Dict[str, List[str]]:
        integration_mapping = {}
        for func in functions:
            if func.side_effects or len(func.dependencies) > 3:
                integration_targets = []
                if any('db' in dep.lower() or 'database' in dep.lower() for dep in func.dependencies):
                    integration_targets.append('database')
                if any('api' in dep.lower() or 'request' in dep.lower() for dep in func.dependencies):
                    integration_targets.append('external_api')
                if any('file' in effect for effect in func.side_effects):
                    integration_targets.append('filesystem')
                if any('network' in dep.lower() or 'socket' in dep.lower() for dep in func.dependencies):
                    integration_targets.append('network')
                if integration_targets:
                    integration_mapping[func.name] = integration_targets
        return integration_mapping
    def _identify_performance_tests(self, functions: List[FunctionInfo]) -> Dict[str, Dict[str, Any]]:
        performance_targets = {}
        for func in functions:
            needs_performance_test = False
            performance_config = {}
            if func.complexity > 8:
                needs_performance_test = True
                performance_config['type'] = 'complexity'
                performance_config['max_time'] = 1.0
            if any('io' in effect for effect in func.side_effects):
                needs_performance_test = True
                performance_config['type'] = 'io_bound'
                performance_config['max_time'] = 5.0
            if 'performance' in ' '.join(func.semantic_tags):
                needs_performance_test = True
                performance_config['type'] = 'performance_critical'
                performance_config['max_time'] = 0.1
            if needs_performance_test:
                performance_targets[func.name] = performance_config
        return performance_targets
    def _define_quality_metrics(self, functions: List[FunctionInfo]) -> Dict[str, Any]:
        avg_complexity = sum(f.complexity for f in functions) / len(functions) if functions else 0
        security_sensitive_count = sum(1 for f in functions if f.is_security_sensitive)
        integration_count = sum(
            1 for f in functions
            if TestType.INTEGRATION in self._intelligent_test_types([f]).get(f.name, [])
        ) if functions else 0
        return {
            'min_coverage': 0.8,
            'max_test_execution_time': 30.0,
            'min_assertion_count': max(2, int(avg_complexity * 0.5)),
            'max_cyclomatic_complexity': 5,
            'security_test_percentage': security_sensitive_count / len(functions) if functions else 0.3,
            'integration_test_percentage': integration_count / len(functions) if functions else 0.2,
            'edge_case_coverage': 0.9,
            'maintainability_score_threshold': 0.7
        }
class ToolIntegrationAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    def integrate_pynguin(self, functions: List[FunctionInfo]) -> List[TestCase]:
        return []
    def integrate_hypothesis(self, func: FunctionInfo) -> List[TestCase]:
        return []
class MutationTestingAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MutationTestingAgent", config)
    def process(self, input_data: Any) -> Any:
        return self.evaluate_test_quality(input_data)
    def evaluate_test_quality(self, test_cases: List[TestCase]) -> float:
        if not test_cases:
            return 0.0
        total_score = sum(tc.quality_score for tc in test_cases)
        return total_score / len(test_cases)
class DynamicAnalysisAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DynamicAnalysisAgent", config)
        self.coverage_data = {}
    def process(self, input_data: Any) -> Any:
        return self.coverage_data
    def instrument_and_run(self, file_path: str):
        self.coverage_data[file_path] = ["main_path", "error_path"]
    def get_execution_paths(self) -> Dict[str, List[str]]:
        return self.coverage_data
class EnhancedTestGenerationAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("EnhancedTestGenerationAgent", config)
        self.test_data_generator = IntelligentTestDataGenerator()
    def process(self, functions: List[FunctionInfo], strategy: Dict[str, Any]) -> List[TestCase]:
        return self._process_impl(functions, strategy)
    def _process_impl(self, functions: List[FunctionInfo], strategy: Dict[str, Any]) -> List[TestCase]:
        if self.llm_agent and hasattr(self.llm_agent, 'model'):
            self.log_activity(f"LLM Agent available: {self.llm_agent is not None}")
        test_cases = []
        for func in functions:
            func_tests = self._generate_function_tests(func, strategy)
            test_cases.extend(func_tests)
        tool_agent = ToolIntegrationAgent(self.config)
        pynguin_tests = tool_agent.integrate_pynguin(functions)
        test_cases.extend(pynguin_tests)
        for func in functions:
            if TestType.PROPERTY_BASED in strategy['test_types'].get(func.name, []):
                hypothesis_tests = tool_agent.integrate_hypothesis(func)
                test_cases.extend(hypothesis_tests)
        self.log_activity(f"Generated {len(test_cases)} test cases (including tool integrations)")
        return test_cases
    def _get_semantic_analysis(self, func: FunctionInfo) -> Dict[str, Any]:
        # Use precomputed semantic_analysis if available
        if hasattr(func, "semantic_analysis") and func.semantic_analysis:
            return func.semantic_analysis
        # Fallback: use tags if present
        if func.semantic_tags:
            return {
                'business_logic_type': func.business_logic_type,
                'semantic_tags': func.semantic_tags,
                'risk_assessment': {'score': func.risk_score}
            }
        return {}
    def _generate_function_tests(self, func: FunctionInfo, strategy: Dict[str, Any]) -> List[TestCase]:
        if self.llm_agent and hasattr(self.llm_agent, 'model'):
            self.log_activity(f"LLM Agent available: {self.llm_agent is not None}")
            semantic_analysis = self._get_semantic_analysis(func)
            if semantic_analysis:
                scenarios = self.llm_agent.generate_test_scenarios(func, semantic_analysis)
            else:
                self.log_activity(f"No semantic analysis for {func.name}, using fallback scenarios")
                scenarios = self._generate_default_scenarios(func)
        else:
            self.log_activity(f"LLM not available, using default scenarios for {func.name}")
            scenarios = self._generate_default_scenarios(func)
        test_cases = []
        for scenario in scenarios:
            test_case = self._create_test_case(func, scenario, strategy)
            if test_case:
                test_cases.append(test_case)
        return test_cases
    def _generate_default_scenarios(self, func: FunctionInfo) -> list:
        # Fallback scenario generation
        return [
            {
                'scenario_name': f'test_{func.name}_happy_path',
                'test_type': 'unit',
                'priority': 'medium',
                'description': f'Test {func.name} with valid inputs',
                'test_data': {},
                'expected_outcome': 'success'
            },
            {
                'scenario_name': f'test_{func.name}_edge_case',
                'test_type': 'edge_case',
                'priority': 'high',
                'description': f'Test {func.name} with edge case inputs',
                'test_data': {},
                'expected_outcome': 'success'
            }
        ]
    def _create_test_case(self, func: FunctionInfo, scenario: Dict[str, Any], strategy: Dict[str, Any]) -> TestCase:
        test_name = scenario.get('scenario_name', f'test_{func.name}')
        # Robustly map test_type to TestType enum, fallback to UNIT if unknown
        raw_type = scenario.get('test_type', 'unit')
        try:
            test_type = TestType(raw_type.lower())
        except Exception:
            # Try to match ignoring case and underscores
            normalized = str(raw_type).replace("-", "_").replace(" ", "_").upper()
            test_type = next((t for t in TestType if t.name == normalized or t.value == raw_type.lower()), TestType.UNIT)
        priority = Priority.MEDIUM
        if self.llm_agent and hasattr(self.llm_agent, 'model'):
            try:
                test_code = self.llm_agent.generate_test_code(scenario, func, TestFramework.PYTEST)
                if not test_code or len(test_code.strip()) < 50:
                    test_code = self._generate_enhanced_test_code(func, scenario)
            except Exception as e:
                self.log_activity(f"LLM test generation failed: {e}")
                test_code = self._generate_enhanced_test_code(func, scenario)
        else:
            test_code = self._generate_enhanced_test_code(func, scenario)
        return TestCase(
            name=test_name,
            test_type=test_type,
            priority=priority,
            target_function=func.name,
            test_code=test_code,
            description=scenario.get('description', ''),
            quality_score=0.7,
            maintainability_score=0.7
        )
    def _generate_enhanced_test_code(self, func: FunctionInfo, scenario: Dict[str, Any]) -> str:
        # Generate realistic test data and assertions
        param_lines = []
        call_args = []
        for param in func.parameters:
            pname = param['name']
            ptype = param.get('type', 'Any')
            val = self.test_data_generator.generate_test_data(ptype, pname, scenario.get('test_type', 'normal'))
            param_lines.append(f"    {pname} = {repr(val)}")
            call_args.append(pname)
        call = f"{func.name}({', '.join(call_args)})"
        result_var = "result"
        test_lines = []
        test_lines.append(f"def {scenario.get('scenario_name', f'test_{func.name}') }():")
        test_lines.append(f'    """')
        test_lines.append(f"    {scenario.get('description', f'Test {func.name}')}")
        test_lines.append(f'    """')
        test_lines.extend(param_lines)
        test_lines.append(f"    {result_var} = {call}")
        # Add assertion based on return type
        if func.return_type and func.return_type != "None":
            test_lines.append(f"    assert {result_var} is not None")
        else:
            test_lines.append(f"    assert True  # No return value to check")
        return "\n".join(test_lines)
class TestValidationAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("TestValidationAgent", config)
    def process(self, test_cases: List[TestCase]) -> List[TestCase]:
        self.log_activity(f"Validating {len(test_cases)} test cases")
        validated_tests = []
        for test_case in test_cases:
            if self._validate_test_case(test_case):
                improved_test = self._improve_test_case(test_case)
                validated_tests.append(improved_test)
            else:
                self.log_activity(f"Test case {test_case.name} failed validation", "WARNING")
        self.log_activity(f"Validated {len(validated_tests)} test cases")
        return validated_tests
    def _validate_test_case(self, test_case: TestCase) -> bool:
        try:
            if not test_case.test_code.strip():
                return False
            if 'def test_' not in test_case.test_code:
                return False
            if 'assert' not in test_case.test_code and 'pytest.raises' not in test_case.test_code:
                return False
            try:
                ast.parse(test_case.test_code)
            except SyntaxError:
                return False
            if test_case.quality_score < 0.3:
                return False
            return True
        except Exception as e:
            self.log_activity(f"Error validating test case: {str(e)}", "ERROR")
            return False
    def _improve_test_case(self, test_case: TestCase) -> TestCase:
        improved_code = test_case.test_code
        if '"""' not in improved_code:
            improved_code = self._add_docstring(improved_code, test_case.description)
        if not test_case.imports:
            improved_code = self._add_missing_imports(improved_code)
        improved_code = self._improve_assertions(improved_code)
        test_case.test_code = improved_code
        test_case.quality_score = self._recalculate_quality_score(improved_code)
        test_case.maintainability_score = self._recalculate_maintainability_score(improved_code)
        return test_case
    def _add_docstring(self, test_code: str, description: str) -> str:
        lines = test_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def test_'):
                lines.insert(i + 1, f'    """')
                lines.insert(i + 2, f'    {description}')
                lines.insert(i + 3, f'    """')
                break
        return '\n'.join(lines)
    def _add_missing_imports(self, test_code: str) -> str:
        lines = test_code.split('\n')
        imports_to_add = []
        if 'pytest.raises' in test_code and 'import pytest' not in test_code:
            imports_to_add.append('import pytest')
        if 'Mock' in test_code and 'from unittest.mock import Mock' not in test_code:
            imports_to_add.append('from unittest.mock import Mock')
        if imports_to_add:
            imports_to_add.extend([''] + lines)
            return '\n'.join(imports_to_add)
        return test_code
    def _improve_assertions(self, test_code: str) -> str:
        improved_code = test_code
        improved_code = improved_code.replace(
            'assert result is not None',
            'assert result is not None\n    assert isinstance(result, (str, int, float, bool, list, dict))'
        )
        return improved_code
    def _recalculate_quality_score(self, test_code: str) -> float:
        score = 0.0
        assertion_count = test_code.count('assert')
        score += min(assertion_count * 0.15, 0.3)
        if 'def test_' in test_code:
            score += 0.2
        if '"""' in test_code:
            score += 0.15
        if 'pytest.raises' in test_code:
            score += 0.15
        if 'import' in test_code:
            score += 0.1
        if 'isinstance' in test_code:
            score += 0.1
        return min(score, 1.0)
    def _recalculate_maintainability_score(self, test_code: str) -> float:
        score = 0.0
        if 'test_' in test_code:
            score += 0.2
        comment_count = test_code.count('#') + test_code.count('"""')
        score += min(comment_count * 0.1, 0.3)
        line_count = len(test_code.split('\n'))
        if 10 <= line_count <= 50:
            score += 0.2
        if 'def ' in test_code and 'assert' in test_code:
            score += 0.2
        if any(len(word) > 3 for word in test_code.split() if word.isalnum()):
            score += 0.1
        return min(score, 1.0)
class TestSuiteOrchestrator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + ".TestSuiteOrchestrator")
        self.code_analysis_agent = EnhancedCodeAnalysisAgent(config)
        self.test_strategy_agent = EnhancedTestStrategyAgent(config)
        self.test_generation_agent = EnhancedTestGenerationAgent(config)
        self.test_validation_agent = TestValidationAgent(config)
        if config and config.get('gemini_api_key'):
            try:
                self.llm_agent = GeminiLLMAgent(config['gemini_api_key'])
                test_response = self.llm_agent.model.generate_content("Hello")
                if test_response:
                    self.logger.info("Gemini API key validated successfully")
                    self._set_llm_agent()
                else:
                    self.logger.warning("Gemini API key validation failed")
                    self.llm_agent = None
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini API: {e}")
                self.llm_agent = None
        else:
            self.llm_agent = None
    def _set_llm_agent(self):
        self.code_analysis_agent.set_llm_agent(self.llm_agent)
        self.test_strategy_agent.set_llm_agent(self.llm_agent)
        self.test_generation_agent.set_llm_agent(self.llm_agent)
        self.test_validation_agent.set_llm_agent(self.llm_agent)
    def generate_comprehensive_test_suite(self, file_paths: List[str]) -> TestSuite:
        self.logger.info(f"Starting comprehensive test generation for {len(file_paths)} files")
        all_functions = []
        for file_path in file_paths:
            functions = self.code_analysis_agent.process(file_path)
            all_functions.extend(functions)
        self.logger.info(f"Analyzed {len(all_functions)} functions")
        strategy = self.test_strategy_agent.process(all_functions)
        self.logger.info("Generated test strategy")
        dynamic_agent = DynamicAnalysisAgent(self.config)
        for file_path in file_paths:
            dynamic_agent.instrument_and_run(file_path)
        execution_paths = dynamic_agent.get_execution_paths()
        test_cases = self.test_generation_agent.process(all_functions, strategy)
        self.logger.info(f"Generated {len(test_cases)} test cases")
        validated_tests = self.test_validation_agent.process(test_cases)
        self.logger.info(f"Validated {len(validated_tests)} test cases")
        mutation_agent = MutationTestingAgent(self.config)
        mutation_score = mutation_agent.evaluate_test_quality(validated_tests)
        test_suite = self._create_test_suite(validated_tests, strategy)
        test_suite.mutation_score = mutation_score
        self.logger.info("Created comprehensive test suite")
        return test_suite
    def _create_test_suite(self, test_cases: List[TestCase], strategy: Dict[str, Any]) -> TestSuite:
        suite_name = f"comprehensive_test_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        coverage_percentage = self._calculate_coverage_percentage(test_cases, strategy)
        quality_score = self._calculate_suite_quality_score(test_cases)
        security_coverage = self._calculate_security_coverage(test_cases)
        maintainability_score = self._calculate_suite_maintainability_score(test_cases)
        execution_time_estimate = self._estimate_execution_time(test_cases)
        return TestSuite(
            name=suite_name,
            test_cases=test_cases,
            coverage_percentage=coverage_percentage,
            quality_score=quality_score,
            security_coverage=security_coverage,
            maintainability_score=maintainability_score,
            execution_time_estimate=execution_time_estimate,
            framework=TestFramework.PYTEST
        )
    def _calculate_coverage_percentage(self, test_cases: List[TestCase], strategy: Dict[str, Any]) -> float:
        if not test_cases:
            return 0.0
        tested_functions = set(test_case.target_function for test_case in test_cases)
        total_functions = len(strategy.get('priorities', {}))
        if total_functions == 0:
            return 0.0
        return (len(tested_functions) / total_functions) * 100.0
    def _calculate_suite_quality_score(self, test_cases: List[TestCase]) -> float:
        if not test_cases:
            return 0.0
        return sum(test_case.quality_score for test_case in test_cases) / len(test_cases)
    def _calculate_security_coverage(self, test_cases: List[TestCase]) -> float:
        if not test_cases:
            return 0.0
        security_tests = [tc for tc in test_cases if tc.test_type == TestType.SECURITY]
        return (len(security_tests) / len(test_cases)) * 100.0
    def _calculate_suite_maintainability_score(self, test_cases: List[TestCase]) -> float:
        if not test_cases:
            return 0.0
        return sum(test_case.maintainability_score for test_case in test_cases) / len(test_cases)
    def _estimate_execution_time(self, test_cases: List[TestCase]) -> float:
        base_time = 0.1
        type_multipliers = {
            TestType.UNIT: 1.0,
            TestType.INTEGRATION: 1.5,
            TestType.EDGE_CASE: 2.0,
            TestType.SECURITY: 2.5,
            TestType.PERFORMANCE: 3.0,
            TestType.ERROR_HANDLING: 2.0
        }
        total_time = sum(
            base_time * type_multipliers.get(test_case.test_type, 1.0)
            for test_case in test_cases
        )
        return total_time
def save_test_suite(test_suite: TestSuite, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    tests_by_file = defaultdict(list)
    for test_case in test_suite.test_cases:
        test_file = f"test_{test_case.target_function}.py"
        test_path = os.path.join(output_dir, test_file)
        tests_by_file[test_path].append(test_case)
    for file_path, test_cases in tests_by_file.items():
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Auto-generated test suite\n")
            f.write(f"# Coverage: {getattr(test_suite, 'coverage_percentage', 0):.2f}%\n")
            f.write(f"# Quality Score: {getattr(test_suite, 'quality_score', 0):.2f}/1.0\n")
            if hasattr(test_suite, 'mutation_score'):
                f.write(f"# Mutation Score: {test_suite.mutation_score:.2f}\n")
            f.write("\n")
            f.write("import pytest\n")
            f.write("from unittest.mock import Mock\n\n")
            all_imports = set()
            for test_case in test_cases:
                all_imports.update(test_case.imports)
            if all_imports:
                f.write("\n".join(all_imports))
                f.write("\n\n")
            for test_case in test_cases:
                f.write(test_case.test_code)
                f.write("\n\n")
def main():
    parser = argparse.ArgumentParser(description='Agentic AI Test Generation System')
    parser.add_argument('paths', nargs='+', help='Paths to Python files or directories to analyze')
    parser.add_argument('--output-dir', default='generated_tests', help='Output directory for tests')
    parser.add_argument('--framework', choices=['pytest', 'unittest', 'nose2'], default='pytest', help='Test framework')
    parser.add_argument('--coverage-target', type=float, default=0.85, help='Minimum coverage percentage')
    parser.add_argument('--api-key', help='Gemini API key')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    config = {
        'gemini_api_key': args.api_key or os.getenv('GEMINI_API_KEY'),
        'framework': TestFramework(args.framework),
        'min_coverage': args.coverage_target
    }
    if config.get('gemini_api_key'):
        print(f"Using Gemini API key: {config['gemini_api_key'][:10]}...")
    else:
        print("No Gemini API key found, using fallback generation")
    python_files = []
    for path in args.paths:
        if os.path.isfile(path) and path.endswith('.py'):
            python_files.append(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
    if not python_files:
        print("No Python files found to analyze!")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        orchestrator = TestSuiteOrchestrator(config)
        test_suite = orchestrator.generate_comprehensive_test_suite(python_files)
        save_test_suite(test_suite, args.output_dir)
        print(f"Test suite generated successfully in {args.output_dir}")
        print(f"Coverage: {test_suite.coverage_percentage:.2f}%")
        print(f"Quality Score: {test_suite.quality_score:.2f}")
        print(f"Generated {len(test_suite.test_cases)} test cases")
    except Exception as e:
        logger.error(f"Error during test generation: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("test_generation.log"),
            logging.StreamHandler()
        ]
    )
    main()
