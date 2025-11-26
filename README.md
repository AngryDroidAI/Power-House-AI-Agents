#!/usr/bin/env python3
"""
Powerhouse Capsule Guild - Ultimate Enhanced Version
- Multi-model orchestration with intelligent fallbacks
- Advanced memory and performance management
- Real-time analytics and quality assurance
- Context-aware processing with conversation memory
- Batch processing and export capabilities
"""

import subprocess
import datetime
import time
import yaml
import csv
import statistics
import json
import hashlib
from pathlib import Path
import argparse
import sys
import os

# === ENHANCED MODEL CONFIG ===
MODELS = {
    "interface": "llama3.1:8b",
    "planner": "mistral:7b", 
    "retriever": "gemma3:12b",
    "summarizer": "deepseek-r1:14b",
    "coder": "qwen3:14b",
    "critic": "phi4:14b",
    "light_critic": "qwen:0.5b",
    "fallback": "tinyllama:1.1b",
    "creative": "llava:7b",  # Added for creative tasks
    "technical": "codellama:13b"  # Added for deep technical tasks
}

# === CONFIG MANAGEMENT ===
class ConfigManager:
    def __init__(self, config_path="~/.powerhouse_config.yaml"):
        self.config_path = Path(config_path).expanduser()
        self.default_config = {
            "models": MODELS,
            "delays": {"agent_delay": 2.0, "retry_delay": 5.0, "batch_delay": 3.0},
            "limits": {"timeout": 180, "max_tokens": 4000, "max_retries": 3, "max_context_items": 5},
            "quality": {"min_length": 50, "max_repetition_ratio": 0.3, "enable_syntax_check": True},
            "memory": {"high_usage_threshold": 75, "light_model_fallback": True, "cache_enabled": True},
            "output": {"save_results": True, "export_format": "both", "max_result_length": 10000}
        }
    
    def load_config(self):
        """Load configuration from YAML file or use defaults"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                    config = self._deep_merge(self.default_config, user_config)
                    print(f"🔧 Configuration loaded from {self.config_path}")
                    return config
        except Exception as e:
            print(f"⚠️ Config load error: {e}, using defaults")
        return self.default_config
    
    def _deep_merge(self, base, user):
        """Recursively merge dictionaries"""
        result = base.copy()
        for key, value in user.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.default_config, f, default_flow_style=False)
            print(f"💾 Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")

# === ENHANCED CONTEXT MANAGER ===
class ContextManager:
    def __init__(self, max_context_items=5):
        self.context = {}
        self.conversation_history = []
        self.working_memory = {}
        self.max_context_items = max_context_items
    
    def add_context(self, agent, content, metadata=None):
        """Store agent output with enhanced metadata"""
        timestamp = datetime.datetime.now()
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        self.context[agent] = {
            'content': content,
            'timestamp': timestamp,
            'metadata': metadata or {},
            'token_estimate': len(content.split()),
            'content_hash': content_hash,
            'quality_score': self._calculate_quality_score(content)
        }
        
        # Maintain only recent items
        if len(self.context) > self.max_context_items:
            oldest_agent = min(self.context.items(), key=lambda x: x[1]['timestamp'])[0]
            del self.context[oldest_agent]
        
        self.conversation_history.append({
            'agent': agent,
            'timestamp': timestamp,
            'content_preview': content[:200] + "..." if len(content) > 200 else content,
            'content_hash': content_hash,
            'duration': metadata.get('duration', 0) if metadata else 0
        })
    
    def _calculate_quality_score(self, content):
        """Calculate a simple quality score based on content characteristics"""
        words = content.split()
        if len(words) < 10:
            return 0.3  # Very short content
        
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words))
        
        # Score based on length, uniqueness, and structure
        length_score = min(len(words) / 100, 1.0)  # Up to 100 words = good
        uniqueness_score = 1.0 - repetition_ratio
        structure_score = 1.0 if any(mark in content for mark in ['\n', '. ', '! ', '? ']) else 0.5
        
        return (length_score + uniqueness_score + structure_score) / 3.0
    
    def get_recent_context(self, max_agents=3, include_quality=False):
        """Get context from recent agents with optional quality info"""
        recent = sorted(self.context.items(), 
                       key=lambda x: x[1]['timestamp'], 
                       reverse=True)[:max_agents]
        
        context_str = ""
        for agent, data in recent:
            quality_info = f" [Quality: {data['quality_score']:.2f}]" if include_quality else ""
            context_str += f"=== {agent}{quality_info} ===\n{data['content'][:500]}...\n\n"
        return context_str.strip()
    
    def get_conversation_summary(self, max_entries=5):
        """Get enhanced summary of recent conversation flow"""
        recent = self.conversation_history[-max_entries:]
        summary = "Recent workflow:\n"
        for entry in recent:
            duration_info = f" ({entry['duration']:.1f}s)" if entry['duration'] > 0 else ""
            summary += f"- {entry['agent']}{duration_info}: {entry['content_preview']}\n"
        return summary
    
    def get_quality_report(self):
        """Generate a quality report for the current context"""
        if not self.context:
            return "No context available for quality analysis."
        
        report = ["Context Quality Report:"]
        total_quality = 0
        
        for agent, data in self.context.items():
            report.append(f"{agent}: Score {data['quality_score']:.2f} "
                         f"(Length: {len(data['content'].split())} words)")
            total_quality += data['quality_score']
        
        avg_quality = total_quality / len(self.context)
        report.append(f"Average Quality Score: {avg_quality:.2f}")
        
        return "\n".join(report)

# === ENHANCED MEMORY MANAGEMENT ===
class MemoryManager:
    @staticmethod
    def check_memory_usage():
        """Check system memory usage percentage with enhanced error handling"""
        try:
            # Try multiple methods to get memory info
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                total = int([line for line in meminfo.splitlines() if "MemTotal" in line][0].split()[1]) // 1024
                available = int([line for line in meminfo.splitlines() if "MemAvailable" in line][0].split()[1]) // 1024
                used = total - available
                return (used / total) * 100
            else:
                # Fallback to free command
                result = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=5)
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    mem = lines[1].split()
                    used_mb = int(mem[2]); total_mb = int(mem[1])
                    return (used_mb / total_mb) * 100
        except Exception as e:
            print(f"⚠️ Memory check error: {e}")
        return 0

    @staticmethod
    def should_use_light_model(threshold=75):
        """Determine if light models should be used due to memory pressure"""
        return MemoryManager.check_memory_usage() > threshold
    
    @staticmethod
    def get_memory_status():
        """Get detailed memory status"""
        usage = MemoryManager.check_memory_usage()
        status = "🟢 Normal" if usage < 50 else "🟡 Moderate" if usage < 75 else "🔴 High"
        return f"{status} ({usage:.1f}% used)"

# === ENHANCED QUALITY ASSURANCE ===
class QualityChecker:
    @staticmethod
    def check_content(content, min_length=50, max_repetition=0.3, enable_syntax_check=False):
        """Comprehensive content quality checks"""
        content = content.strip()
        
        # Basic checks
        if len(content) < min_length:
            return False, f"Content too short ({len(content)} chars, minimum {min_length})"
        
        words = content.split()
        if len(words) < 5:
            return False, "Content has too few words"
        
        # Repetition check
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words))
        if repetition_ratio > max_repetition:
            return False, f"High repetition detected ({repetition_ratio:.1%})"
        
        # Error pattern detection
        error_indicators = ["error:", "timeout", "failed", "❌", "⏰", "exception", "traceback"]
        if any(indicator in content.lower() for indicator in error_indicators):
            return False, "Content contains error indicators"
        
        # Structure checks
        if len(content) > 100 and not any(mark in content for mark in ['. ', '! ', '? ', '\n']):
            return False, "Content lacks proper sentence structure"
        
        # Optional syntax check for code-like content
        if enable_syntax_check and QualityChecker.looks_like_code(content):
            syntax_ok, syntax_msg = QualityChecker.check_python_syntax(content)
            if not syntax_ok:
                return False, f"Code syntax issue: {syntax_msg}"
        
        return True, "Quality check passed"
    
    @staticmethod
    def looks_like_code(content):
        """Heuristic to detect code-like content"""
        code_indicators = ['def ', 'class ', 'import ', 'function ', 'var ', 'const ', '<?php', '<html', '#include']
        return any(indicator in content.lower() for indicator in code_indicators)
    
    @staticmethod
    def check_python_syntax(content):
        """Basic Python syntax check using ast"""
        try:
            # Simple check for common Python patterns without executing
            lines = content.split('\n')
            indent_level = 0
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                
                # Basic indentation check
                if stripped.startswith('def ') or stripped.startswith('class ') or stripped.startswith('if ') or stripped.startswith('for '):
                    if not line.startswith(' ' * indent_level):
                        return False, f"Indentation error near: {stripped[:50]}"
                    indent_level += 4
                elif line.rstrip().endswith(':'):
                    indent_level += 4
                elif stripped and line.startswith(' ' * (indent_level - 4)) and not line.startswith(' ' * indent_level):
                    indent_level -= 4
            
            return True, "Basic syntax appears valid"
        except Exception as e:
            return False, f"Syntax check error: {e}"

# === RESULT MANAGER ===
class ResultManager:
    def __init__(self, output_dir="~/capsule_results"):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(exist_ok=True)
    
    def save_result(self, task, result, context_manager, format_type="both"):
        """Save result in specified format(s)"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task_hash = hashlib.md5(task.encode()).hexdigest()[:8]
        filename_base = f"result_{timestamp}_{task_hash}"
        
        try:
            if format_type in ["txt", "both"]:
                txt_file = self.output_dir / f"{filename_base}.txt"
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(f"Task: {task}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Result Hash: {task_hash}\n")
                    f.write("="*50 + "\n")
                    f.write(result)
                    f.write("\n" + "="*50 + "\n")
                    f.write("\nContext Summary:\n")
                    f.write(context_manager.get_conversation_summary())
                print(f"💾 Result saved to {txt_file}")
            
            if format_type in ["json", "both"]:
                json_file = self.output_dir / f"{filename_base}.json"
                result_data = {
                    "task": task,
                    "timestamp": timestamp,
                    "result_hash": task_hash,
                    "result": result,
                    "context_summary": context_manager.get_conversation_summary(),
                    "quality_report": context_manager.get_quality_report(),
                    "performance_metrics": {
                        "memory_usage": MemoryManager.check_memory_usage(),
                        "context_items": len(context_manager.context)
                    }
                }
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                print(f"💾 JSON result saved to {json_file}")
                
            return True
        except Exception as e:
            print(f"❌ Error saving result: {e}")
            return False

# === PERFORMANCE TRACKING ===
LOG_DIR = Path.home() / "capsule_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "powerhouse_guild.log"
PERFORMANCE_LOG = LOG_DIR / "performance.csv"
CACHE_DIR = LOG_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

def log_performance(agent, duration, success=True, retries=0, model_used="", quality_score=0.0):
    """Enhanced performance logging with quality metrics"""
    try:
        if not PERFORMANCE_LOG.exists():
            with open(PERFORMANCE_LOG, "w") as f:
                f.write("timestamp,agent,duration_seconds,success,retries,model_used,memory_usage,quality_score\n")
        
        memory_usage = MemoryManager.check_memory_usage()
        with open(PERFORMANCE_LOG, "a") as f:
            ts = datetime.datetime.now().isoformat()
            f.write(f"{ts},{agent},{duration:.2f},{success},{retries},{model_used},{memory_usage:.1f},{quality_score:.2f}\n")
    except Exception as e:
        print(f"⚠️ Could not log performance: {e}")

def log_event(agent, message, level="INFO"):
    """Enhanced logging with levels and context"""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory_usage = MemoryManager.check_memory_usage()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{ts} | {level} | MEM:{memory_usage:.1f}%] {agent}: {message}\n")
    except:
        pass

# === CACHE SYSTEM ===
class CacheSystem:
    @staticmethod
    def get_cache_key(prompt, agent):
        """Generate cache key from prompt and agent"""
        content = f"{agent}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    @staticmethod
    def get_cached_result(cache_key, max_age_minutes=60):
        """Get cached result if available and fresh"""
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            
            cache_time = datetime.datetime.fromisoformat(cache_data['timestamp'])
            age_minutes = (datetime.datetime.now() - cache_time).total_seconds() / 60
            
            if age_minutes <= max_age_minutes:
                return cache_data['result']
            else:
                # Delete expired cache
                cache_file.unlink()
                return None
        except:
            return None
    
    @staticmethod
    def cache_result(cache_key, result):
        """Cache result with timestamp"""
        cache_file = CACHE_DIR / f"{cache_key}.json"
        try:
            cache_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'result': result
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
            return True
        except:
            return False

# === ENHANCED AGENT RUNNER ===
def run_agent_with_fallback(model, prompt, role, config, context_manager=None, timeout=180, max_retries=3):
    """Enhanced agent runner with cache and improved error handling"""
    # Check cache first if enabled
    if config['memory']['cache_enabled']:
        cache_key = CacheSystem.get_cache_key(prompt, role)
        cached_result = CacheSystem.get_cached_result(cache_key)
        if cached_result:
            log_event(role, f"CACHE HIT | Using cached result")
            print(f"💾 {role} using cached result")
            return cached_result
    
    original_model = model
    retries = 0
    
    # Check memory pressure
    if (config['memory']['light_model_fallback'] and 
        MemoryManager.should_use_light_model(config['memory']['high_usage_threshold'])):
        if "14b" in model or "13b" in model:
            if role == "critic":
                model = config['models']['light_critic']
            elif role in ["summarizer", "coder"]:
                model = config['models']['fallback']
            log_event(role, f"Memory high: using lighter model ({model})")
    
    while retries <= max_retries:
        start_time = time.time()
        current_model = model if retries == 0 else config['models']['fallback']
        
        log_event(role, f"ATTEMPT {retries+1} | Model: {current_model}")
        print(f"⚡ {role} attempt {retries+1} with {current_model}...")
        
        try:
            result = subprocess.run(
                ["ollama", "run", current_model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                output = result.stdout.decode("utf-8", errors="ignore").strip()
                
                # Enhanced quality check
                quality_ok, quality_msg = QualityChecker.check_content(
                    output, 
                    config['quality']['min_length'],
                    config['quality']['max_repetition_ratio'],
                    config['quality']['enable_syntax_check']
                )
                
                if quality_ok:
                    # Calculate quality score
                    quality_score = len(output.split()) / 100  # Simple score based on length
                    quality_score = min(quality_score, 1.0)
                    
                    log_event(role, f"SUCCESS ({duration:.1f}s) | Quality: {quality_score:.2f}")
                    log_performance(role, duration, True, retries, current_model, quality_score)
                    
                    # Cache the result
                    if config['memory']['cache_enabled']:
                        CacheSystem.cache_result(cache_key, output)
                    
                    # Store in context
                    if context_manager:
                        context_manager.add_context(role, output, {
                            'model': current_model,
                            'duration': duration,
                            'retries': retries,
                            'quality_score': quality_score
                        })
                    
                    print(f"✅ {role} completed in {duration:.1f}s (attempt {retries+1})")
                    return output
                else:
                    log_event(role, f"QUALITY ISSUE: {quality_msg}")
                    # Fall through to retry logic
            else:
                err = result.stderr.decode("utf-8", errors="ignore").strip()
                log_event(role, f"ERROR: {err}")
        
        except subprocess.TimeoutExpired:
            log_event(role, f"TIMEOUT after {timeout}s")
        
        except Exception as e:
            log_event(role, f"EXCEPTION: {e}")
        
        # Exponential backoff for retries
        retries += 1
        if retries <= max_retries:
            retry_delay = config['delays']['retry_delay'] * (2 ** (retries - 1))  # Exponential backoff
            log_event(role, f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            model = config['models']['fallback']  # Use fallback for retries
    
    # All attempts failed
    duration = time.time() - start_time
    log_performance(role, duration, False, retries, current_model, 0.0)
    return f"❌ {role} failed after {retries} attempts"

# === ENHANCED AGENT PROMPTS ===
def interface_agent(user_input, config, context_manager=None):
    recent_context = context_manager.get_recent_context(2) if context_manager else ""
    prompt = f"""Refine and enhance this user query for optimal processing:

USER INPUT: {user_input}

RECENT CONTEXT:
{recent_context}

Provide a clear, well-structured, and context-aware version that will guide subsequent agents effectively. Consider the conversation flow and maintain coherence."""
    return run_agent_with_fallback(config['models']['interface'], prompt, "🎯 Interface", config, context_manager)

def planner_agent(task, context="", config=None, context_manager=None):
    recent_context = context_manager.get_recent_context(3, include_quality=True) if context_manager else ""
    prompt = f"""Create a comprehensive execution plan for this task.

TASK: {task}
ADDITIONAL CONTEXT: {context}
RECENT CONTEXT (with quality scores):
{recent_context}

Break down the task into logical steps, suggest the optimal approach, and identify potential challenges. Consider the quality of available context."""
    return run_agent_with_fallback(config['models']['planner'], prompt, "📋 Planner", config, context_manager)

def retriever_agent(query, plan="", config=None, context_manager=None):
    prompt = f"""Retrieve comprehensive, high-quality context information for this query.

QUERY: {query}
EXECUTION PLAN: {plan}

Provide 3-5 well-chosen, detailed context chunks that would help answer the query thoroughly. Focus on relevance, accuracy, and comprehensiveness."""
    return run_agent_with_fallback(config['models']['retriever'], prompt, "🔍 Retriever", config, context_manager)

def summarizer_agent(text, focus="", config=None, context_manager=None):
    prompt = f"""Create a nuanced, insightful summary focusing on the specified aspect.

FOCUS: {focus}
TEXT TO SUMMARIZE: {text}

Provide a comprehensive yet concise summary that captures key insights, maintains context, and addresses the core query effectively."""
    return run_agent_with_fallback(config['models']['summarizer'], prompt, "📝 Summarizer", config, context_manager)

def coder_agent(spec, language_hint="", config=None, context_manager=None):
    prompt = f"""Write professional-grade, production-ready code based on this specification.

SPECIFICATION: {spec}
LANGUAGE HINT: {language_hint}

Ensure the code is efficient, well-commented, follows best practices, includes error handling, and is ready for deployment."""
    return run_agent_with_fallback(config['models']['coder'], prompt, "💻 Coder", config, context_manager)

def critic_agent(content, criteria="comprehensive analysis", config=None, context_manager=None):
    prompt = f"""Provide a detailed, constructive critique focusing on {criteria}.

CONTENT TO CRITIQUE: {content}

Offer specific, actionable feedback highlighting strengths, identifying areas for improvement, and suggesting concrete enhancements."""
    return run_agent_with_fallback(config['models']['critic'], prompt, "🔎 Critic", config, context_manager)

def creative_agent(task, style_hint="", config=None, context_manager=None):
    prompt = f"""Generate creative, engaging content for this task.

TASK: {task}
STYLE HINT: {style_hint}

Create original, imaginative content that is engaging, well-structured, and appropriate for the specified context."""
    return run_agent_with_fallback(config['models']['creative'], prompt, "🎨 Creative", config, context_manager)

# === ENHANCED TASK TYPE ANALYSIS ===
def analyze_task_type(user_input):
    lower = user_input.lower()
    code_keywords = ["code", "program", "script", "function", "python", "html", "css", "javascript", "bash", "sql", "algorithm"]
    complex_keywords = ["analyze", "compare", "critique", "evaluate", "design", "philosophy", "ethics", "strategy", "framework", "research"]
    creative_keywords = ["creative", "story", "poem", "art", "design", "imagine", "narrative", "fiction"]
    technical_keywords = ["technical", "architecture", "system", "infrastructure", "optimize", "performance", "scalability"]
    
    if any(kw in lower for kw in creative_keywords):
        return "creative"
    elif any(kw in lower for kw in technical_keywords):
        return "technical"
    elif any(kw in lower for kw in code_keywords):
        return "code"
    elif any(kw in lower for kw in complex_keywords):
        return "complex"
    else:
        return "general"

# === ULTIMATE ENHANCED PIPELINE ===
def powerhouse_capsule_pipeline(user_input, config, result_manager=None):
    """Ultimate enhanced pipeline with intelligent routing and quality control"""
    print("🚀 Starting Ultimate Powerhouse Capsule Guild...")
    context_manager = ContextManager(config['limits']['max_context_items'])
    
    task_type = analyze_task_type(user_input)
    print(f"🎯 Task type: {task_type}")
    
    # Track comprehensive metrics
    start_memory = MemoryManager.check_memory_usage()
    start_time = time.time()
    log_event("SYSTEM", f"Pipeline start | Memory: {start_memory:.1f}% | Task: {task_type} | Input: '{user_input[:100]}...'")
    
    # Enhanced agent execution with intelligent routing
    refined = interface_agent(user_input, config, context_manager)
    time.sleep(config['delays']['agent_delay'])
    
    plan = planner_agent(refined, f"Task type: {task_type}", config, context_manager)
    time.sleep(config['delays']['agent_delay'])
    
    # Task-specific processing pipelines
    if task_type == "code":
        code = coder_agent(refined, "", config, context_manager)
        time.sleep(config['delays']['agent_delay'])
        
        critique = critic_agent(code, "code quality, efficiency, security, maintainability", config, context_manager)
        time.sleep(config['delays']['agent_delay'])
        
        final = interface_agent(f"Code Implementation:\n{code}\n\nComprehensive Review:\n{critique}", config, context_manager)
    
    elif task_type == "creative":
        creative_content = creative_agent(refined, "innovative and engaging", config, context_manager)
        time.sleep(config['delays']['agent_delay'])
        
        critique = critic_agent(creative_content, "creativity, coherence, engagement, style", config, context_manager)
        time.sleep(config['delays']['agent_delay'])
        
        final = interface_agent(f"Creative Content:\n{creative_content}\n\nRefinement Analysis:\n{critique}", config, context_manager)
    
    elif task_type == "technical":
        # Technical tasks get deeper analysis
        context = retriever_agent(refined, plan, config, context_manager)
        time.sleep(config['delays']['agent_delay'])
        
        technical_analysis = summarizer_agent(context, "technical details and implications", config, context_manager)
        time.sleep(config['delays']['agent_delay'])
        
        deep_critique = critic_agent(technical_analysis, "technical accuracy, scalability, implementation feasibility", config, context_manager)
        time.sleep(config['delays']['agent_delay'])
        
        final = interface_agent(f"Technical Analysis:\n{technical_analysis}\n\nExpert Review:\n{deep_critique}", config, context_manager)
    
    else:  # general or complex
        context = retriever_agent(refined, plan, config, context_manager)
        time.sleep(config['delays']['agent_delay'])
        
        summary = summarizer_agent(context, "comprehensive answer to the query", config, context_manager)
        time.sleep(config['delays']['agent_delay'])
        
        critique = critic_agent(summary, "accuracy, completeness, clarity, depth", config, context_manager)
        time.sleep(config['delays']['agent_delay'])
        
        final = interface_agent(f"Comprehensive Analysis:\n{summary}\n\nQuality Review:\n{critique}", config, context_manager)
    
    # Final quality assurance
    quality_ok, quality_msg = QualityChecker.check_content(final, 
                                                         config['quality']['min_length'],
                                                         config['quality']['max_repetition_ratio'])
    
    if not quality_ok:
        log_event("SYSTEM", f"Final output quality issue: {quality_msg}")
        # Attempt one refinement pass
        final = interface_agent(f"Original: {final}\n\nIssue: {quality_msg}\nPlease improve quality:", config, context_manager)
    
    # Calculate overall performance metrics
    end_time = time.time()
    total_duration = end_time - start_time
    end_memory = MemoryManager.check_memory_usage()
    memory_delta = end_memory - start_memory
    
    # Save result if configured
    if result_manager and config['output']['save_results']:
        result_manager.save_result(user_input, final, context_manager, config['output']['export_format'])
    
    # Comprehensive completion report
    completion_report = f"""
=== PIPELINE COMPLETION REPORT ===
Task: {user_input[:100]}...
Type: {task_type}
Duration: {total_duration:.1f}s
Memory: {start_memory:.1f}% → {end_memory:.1f}% (Δ: {memory_delta:+.1f}%)
Context Items: {len(context_manager.context)}
Final Quality: {"✅ Good" if quality_ok else "⚠️ Needs review"}
{context_manager.get_quality_report()}
"""
    
    log_event("SYSTEM", f"Pipeline complete | Duration: {total_duration:.1f}s | Memory Δ: {memory_delta:+.1f}%")
    print(completion_report)
    
    return final, context_manager, completion_report

# === ENHANCED BATCH PROCESSING ===
def batch_process_agents(tasks, config):
    """Advanced batch processing with progress tracking and results export"""
    print(f"🔄 Advanced batch processing {len(tasks)} tasks...")
    result_manager = ResultManager()
    results = {}
    successful_tasks = 0
    failed_tasks = 0
    
    for i, task in enumerate(tasks, 1):
        print(f"\n--- Processing Task {i}/{len(tasks)}: '{task[:50]}...' ---")
        try:
            result, context, report = powerhouse_capsule_pipeline(task, config, result_manager)
            results[task] = {
                'result': result,
                'context': context.get_conversation_summary(),
                'report': report,
                'status': 'success'
            }
            successful_tasks += 1
            
            # Enhanced progress display
            progress = (i / len(tasks)) * 100
            print(f"📊 Progress: {progress:.1f}% | Successful: {successful_tasks} | Failed: {failed_tasks}")
            
        except Exception as e:
            error_msg = f"Batch processing error: {e}"
            results[task] = {'error': error_msg, 'status': 'failed'}
            failed_tasks += 1
            log_event("BATCH", f"Task failed: {task} | Error: {e}")
        
        # Adaptive delay between tasks
        if i < len(tasks):
            current_memory = MemoryManager.check_memory_usage()
            delay_multiplier = 1.0
            if current_memory > 80:
                delay_multiplier = 2.0  # Longer delay if memory is high
            elif current_memory > 60:
                delay_multiplier = 1.5
            
            delay = config['delays']['batch_delay'] * delay_multiplier
            print(f"⏳ Next task in {delay:.1f}s (memory: {current_memory:.1f}%)...")
            time.sleep(delay)
    
    # Batch completion report
    batch_report = f"""
=== BATCH PROCESSING COMPLETE ===
Total Tasks: {len(tasks)}
Successful: {successful_tasks}
Failed: {failed_tasks}
Success Rate: {(successful_tasks/len(tasks))*100:.1f}%
Results saved to: {result_manager.output_dir}
"""
    print(batch_report)
    
    return results, batch_report

# === ULTIMATE PERFORMANCE ANALYTICS ===
def generate_performance_report(detailed=False):
    """Generate comprehensive performance analysis without pandas"""
    if not PERFORMANCE_LOG.exists():
        return "📊 No performance data available yet."
    
    try:
        # Read CSV data manually
        with open(PERFORMANCE_LOG, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return "📊 Performance log is empty."
        
        report = []
        report.append("=== ULTIMATE POWERHOUSE CAPSULE GUILD PERFORMANCE REPORT ===")
        
        # Extract dates from timestamps
        timestamps = [row['timestamp'] for row in rows]
        min_date = min(timestamps)[:10] if timestamps else "N/A"
        max_date = max(timestamps)[:10] if timestamps else "N/A"
        
        report.append(f"📅 Period: {min_date} to {max_date}")
        report.append(f"📈 Total runs: {len(rows)}")
        
        # Calculate overall statistics
        durations = [float(row['duration_seconds']) for row in rows]
        success_count = sum(1 for row in rows if row['success'] == 'True')
        memory_usage = [float(row['memory_usage']) for row in rows if row['memory_usage']]
        quality_scores = [float(row['quality_score']) for row in rows if row['quality_score']]
        
        success_rate = (success_count / len(rows)) * 100 if rows else 0
        avg_duration = statistics.mean(durations) if durations else 0
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        report.append(f"✅ Success rate: {success_rate:.1f}%")
        report.append(f"⏱️  Average duration: {avg_duration:.1f}s")
        report.append(f"💾 Average memory usage: {avg_memory:.1f}%")
        report.append(f"⭐ Average quality score: {avg_quality:.2f}")
        
        # Group by agent for detailed analysis
        agents_data = {}
        for row in rows:
            agent = row['agent']
            if agent not in agents_data:
                agents_data[agent] = []
            agents_data[agent].append(row)
        
        report.append("\n--- AGENT PERFORMANCE ANALYSIS ---")
        
        for agent, data in sorted(agents_data.items()):
            agent_durations = [float(row['duration_seconds']) for row in data]
            agent_success_count = sum(1 for row in data if row['success'] == 'True')
            agent_retries = [int(row['retries']) for row in data]
            agent_quality = [float(row['quality_score']) for row in data if row['quality_score']]
            
            agent_success_rate = (agent_success_count / len(data)) * 100
            avg_agent_duration = statistics.mean(agent_durations) if agent_durations else 0
            std_agent_duration = statistics.stdev(agent_durations) if len(agent_durations) > 1 else 0
            avg_agent_retries = statistics.mean(agent_retries) if agent_retries else 0
            avg_agent_quality = statistics.mean(agent_quality) if agent_quality else 0
            
            report.append(
                f"{agent}: {avg_agent_duration:.1f}s (±{std_agent_duration:.1f}s) | "
                f"Success: {agent_success_rate:.1f}% | "
                f"Retries: {avg_agent_retries:.1f} | "
                f"Quality: {avg_agent_quality:.2f} | "
                f"Runs: {len(data)}"
            )
        
        # Advanced recommendations
        report.append("\n--- INTELLIGENT RECOMMENDATIONS ---")
        
        # Model optimization suggestions
        slow_agents = [agent for agent, data in agents_data.items() 
                      if statistics.mean([float(row['duration_seconds']) for row in data]) > 30]
        if slow_agents:
            report.append(f"🚀 Consider lighter models for: {', '.join(slow_agents)}")
        
        high_retry_agents = [agent for agent, data in agents_data.items()
                           if statistics.mean([int(row['retries']) for row in data]) > 0.5]
        if high_retry_agents:
            report.append(f"🔧 Improve reliability for: {', '.join(high_retry_agents)}")
        
        low_quality_agents = [agent for agent, data in agents_data.items()
                            if statistics.mean([float(row['quality_score']) for row in data if row['quality_score']]) < 0.5]
        if low_quality_agents:
            report.append(f"📊 Enhance quality for: {', '.join(low_quality_agents)}")
        
        # Performance trends
        recent_rows = rows[-10:]  # Last 10 runs
        if len(recent_rows) >= 5:
            recent_durations = [float(row['duration_seconds']) for row in recent_rows]
            recent_quality = [float(row['quality_score']) for row in recent_rows if row['quality_score']]
            
            trend_duration = "improving" if statistics.mean(recent_durations) < avg_duration else "declining"
            trend_quality = "improving" if statistics.mean(recent_quality) > avg_quality else "declining"
            
            report.append(f"📈 Recent trends: Duration {trend_duration}, Quality {trend_quality}")
        
        # System health check
        current_memory = MemoryManager.check_memory_usage()
        memory_status = "🟢 Healthy" if current_memory < 70 else "🟡 Moderate" if current_memory < 85 else "🔴 Critical"
        report.append(f"❤️  System health: {memory_status} ({current_memory:.1f}% memory used)")
        
        return "\n".join(report)
    
    except Exception as e:
        return f"❌ Error generating report: {e}"

# === INTERACTIVE MODE ===
def interactive_mode(config):
    print("=== ULTIMATE POWERHOUSE CAPSULE GUILD ===")
    print("💡 Commands: 'quit', 'models', 'status', 'report', 'config', 'context', 'quality', 'saveconfig'")
    print("💾 Enhanced context preservation with quality tracking")
    
    result_manager = ResultManager()
    session_context = ContextManager(config['limits']['max_context_items'])
    
    while True:
        try:
            user_input = input("\n🎯 Enter task or command: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'models':
                for role, model in config['models'].items():
                    print(f"{role:15} → {model}")
                continue
            elif user_input.lower() == 'status':
                mem_status = MemoryManager.get_memory_status()
                print(f"📈 {mem_status}")
                print(f"📋 Logs: {LOG_FILE}")
                print(f"📊 Performance: {PERFORMANCE_LOG}")
                if session_context.conversation_history:
                    print("\nRecent context:")
                    print(session_context.get_conversation_summary(3))
                continue
            elif user_input.lower() == 'report':
                print(generate_performance_report(detailed=True))
                continue
            elif user_input.lower() == 'config':
                print("Current configuration:")
                for key, value in config.items():
                    print(f"{key}: {value}")
                continue
            elif user_input.lower() == 'context':
                if session_context.context:
                    print("Current context (with quality scores):")
                    print(session_context.get_recent_context(include_quality=True))
                else:
                    print("No context available yet.")
                continue
            elif user_input.lower() == 'quality':
                if session_context.context:
                    print(session_context.get_quality_report())
                else:
                    print("No context available for quality analysis.")
                continue
            elif user_input.lower() == 'saveconfig':
                config_manager = ConfigManager()
                config_manager.save_config()
                continue
            elif not user_input:
                continue
                
            # Process the task
            result, context, report = powerhouse_capsule_pipeline(user_input, config, result_manager)
            session_context = context  # Update session context
            
            print("\n" + "="*60)
            print("ENHANCED FINAL RESULT")
            print("="*60)
            print(result)
            print("="*60)
            print(f"\n💡 Tip: Use 'context' or 'quality' commands to see detailed analysis")
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Session interrupted. Type 'quit' to exit or continue with new task.")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            log_event("INTERACTIVE", f"Error: {e}", "ERROR")

# === MAIN EXECUTION ===
def main():
    """Ultimate main function with enhanced command-line options"""
    parser = argparse.ArgumentParser(description='Ultimate Powerhouse Capsule Guild')
    parser.add_argument('--batch', help='Process multiple tasks from file (one per line)')
    parser.add_argument('--report', action='store_true', help='Generate detailed performance report')
    parser.add_argument('--config', help='Custom config file path')
    parser.add_argument('--task', help='Single task to process (non-interactive)')
    parser.add_argument('--export', choices=['txt', 'json', 'both'], default='both', help='Export format for results')
    parser.add_argument('--max-retries', type=int, help='Override maximum retries')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config if args.config else "~/.powerhouse_config.yaml")
    config = config_manager.load_config()
    
    # Apply command-line overrides
    if args.export:
        config['output']['export_format'] = args.export
    if args.max_retries:
        config['limits']['max_retries'] = args.max_retries
    if args.no_cache:
        config['memory']['cache_enabled'] = False
    
    print(f"🔧 Loaded enhanced configuration with {len(config['models'])} models")
    print(f"💾 Caching: {'Enabled' if config['memory']['cache_enabled'] else 'Disabled'}")
    print(f"📊 Export format: {config['output']['export_format']}")
    
    # Handle command-line options
    if args.report:
        print(generate_performance_report(detailed=True))
        return
    elif args.batch:
        try:
            with open(args.batch, 'r') as f:
                tasks = [line.strip() for line in f if line.strip()]
            results, batch_report = batch_process_agents(tasks, config)
            print(batch_report)
            return
        except FileNotFoundError:
            print(f"❌ Batch file not found: {args.batch}")
            return
    elif args.task:
        # Single task mode with enhanced output
        result_manager = ResultManager()
        result, context, report = powerhouse_capsule_pipeline(args.task, config, result_manager)
        print("\n=== ENHANCED RESULT ===")
        print(result)
        print("\n=== PERFORMANCE REPORT ===")
        print(report)
        return
    else:
        # Interactive mode
        interactive_mode(config)

if __name__ == "__main__":
    main()
