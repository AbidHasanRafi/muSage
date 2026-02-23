"""
Minimal GPT implementation for μSage - inspired by @karpathy's atomic GPT.
Pure Python, zero dependencies, lightweight Q&A model.
"""

import math
import random
import pickle
import os

# ============================================================================
# Autograd Engine
# ============================================================================

class Value:
    """Scalar value with automatic differentiation"""
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))
    
    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))
    
    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    
    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """Backpropagate gradients through computation graph"""
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ============================================================================
# Model Architecture
# ============================================================================

def linear(x, w):
    """Linear layer: x @ w.T"""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """Softmax with numerical stability"""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """Root mean square normalization"""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt_forward(token_id, pos_id, keys, values, state_dict, config):
    """Single forward pass through the GPT model"""
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(config['n_layer']):
        # Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        
        x_attn = []
        n_head = config['n_head']
        head_dim = config['head_dim']
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 
                          for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) 
                       for j in range(head_dim)]
            x_attn.extend(head_out)
        
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        
        # MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# ============================================================================
# Tokenizer
# ============================================================================

class CharTokenizer:
    """Character-level tokenizer"""
    
    def __init__(self, vocab=None):
        if vocab is None:
            # Default vocab: lowercase + space + punctuation
            vocab = list("abcdefghijklmnopqrstuvwxyz0123456789 .,?!'-:")
        self.vocab = vocab
        self.char_to_id = {ch: i for i, ch in enumerate(vocab)}
        self.BOS = len(vocab)  # Beginning of sequence
        self.EOS = len(vocab) + 1  # End of sequence
        self.vocab_size = len(vocab) + 2
    
    def encode(self, text):
        """Convert text to token IDs"""
        text = text.lower()
        tokens = [self.BOS]
        for ch in text:
            if ch in self.char_to_id:
                tokens.append(self.char_to_id[ch])
        tokens.append(self.EOS)
        return tokens
    
    def decode(self, tokens):
        """Convert token IDs to text"""
        chars = []
        for token_id in tokens:
            if token_id == self.BOS or token_id == self.EOS:
                continue
            if 0 <= token_id < len(self.vocab):
                chars.append(self.vocab[token_id])
        return ''.join(chars)

# ============================================================================
# MiniGPT Model
# ============================================================================

class MiniGPT:
    """Minimal GPT for Q&A generation"""
    
    def __init__(self, config=None):
        if config is None:
            config = {
                'n_layer': 2,       # 2 transformer layers
                'n_embd': 32,       # embedding dimension
                'block_size': 128,  # max context length
                'n_head': 4,        # attention heads
            }
        config['head_dim'] = config['n_embd'] // config['n_head']
        self.config = config
        self.tokenizer = CharTokenizer()
        self.state_dict = None
        self.trained = False
    
    def _init_params(self):
        """Initialize model parameters"""
        random.seed(42)
        vocab_size = self.tokenizer.vocab_size
        n_embd = self.config['n_embd']
        n_layer = self.config['n_layer']
        block_size = self.config['block_size']
        
        def matrix(nout, nin, std=0.08):
            return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
        
        state_dict = {
            'wte': matrix(vocab_size, n_embd),
            'wpe': matrix(block_size, n_embd),
            'lm_head': matrix(vocab_size, n_embd)
        }
        
        for i in range(n_layer):
            state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
            state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
            state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
            state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
            state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
            state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
        
        self.state_dict = state_dict
        self.params = [p for mat in state_dict.values() for row in mat for p in row]
    
    def train(self, qa_pairs, num_steps=2000, learning_rate=0.01):
        """Train on Q&A pairs: list of (question, answer) tuples"""
        if self.state_dict is None:
            self._init_params()
        
        # Format training data: "Q: question A: answer"
        docs = [f"Q: {q} A: {a}" for q, a in qa_pairs]
        random.shuffle(docs)
        
        # Adam optimizer buffers
        beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
        m = [0.0] * len(self.params)
        v = [0.0] * len(self.params)
        
        print(f"Training MiniGPT on {len(docs)} Q&A pairs...")
        
        for step in range(num_steps):
            # Get training example
            doc = docs[step % len(docs)]
            tokens = self.tokenizer.encode(doc)
            n = min(self.config['block_size'], len(tokens) - 1)
            
            # Forward pass
            keys = [[] for _ in range(self.config['n_layer'])]
            values = [[] for _ in range(self.config['n_layer'])]
            losses = []
            
            for pos_id in range(n):
                token_id = tokens[pos_id]
                target_id = tokens[pos_id + 1]
                logits = gpt_forward(token_id, pos_id, keys, values, self.state_dict, self.config)
                probs = softmax(logits)
                loss_t = -probs[target_id].log()
                losses.append(loss_t)
            
            loss = (1 / n) * sum(losses)
            
            # Backward pass
            loss.backward()
            
            # Adam update with learning rate decay
            lr_t = learning_rate * (1 - step / num_steps)
            for i, p in enumerate(self.params):
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
                m_hat = m[i] / (1 - beta1 ** (step + 1))
                v_hat = v[i] / (1 - beta2 ** (step + 1))
                p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
                p.grad = 0
            
            if (step + 1) % 50 == 0 or step < 10:
                print(f"Step {step+1}/{num_steps} | Loss: {loss.data:.4f}")
        
        self.trained = True
        print("Training complete!")
    
    def generate(self, prompt, max_length=100, temperature=0.5):
        """Generate answer given a question"""
        if not self.trained or self.state_dict is None:
            return "Model not trained yet."
        
        # Encode prompt
        prompt_text = f"Q: {prompt} A:"
        tokens = self.tokenizer.encode(prompt_text)
        
        # Generate completion
        keys = [[] for _ in range(self.config['n_layer'])]
        values = [[] for _ in range(self.config['n_layer'])]
        
        for pos_id, token_id in enumerate(tokens[:-1]):  # Skip EOS
            _ = gpt_forward(token_id, pos_id, keys, values, self.state_dict, self.config)
        
        # Generate new tokens
        generated = []
        pos_id = len(tokens) - 1
        
        for _ in range(max_length):
            if pos_id >= self.config['block_size']:
                break
            
            token_id = tokens[-1] if tokens else self.tokenizer.BOS
            logits = gpt_forward(token_id, pos_id, keys, values, self.state_dict, self.config)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(self.tokenizer.vocab_size), 
                                    weights=[p.data for p in probs])[0]
            
            if token_id == self.tokenizer.EOS:
                break
            
            tokens.append(token_id)
            generated.append(token_id)
            pos_id += 1
        
        # Decode answer (skip the prompt part)
        answer = self.tokenizer.decode(generated)
        return answer.strip()
    
    def save(self, path):
        """Save model weights to disk"""
        # Extract raw data from Value objects
        state_dict_data = {}
        for key, mat in self.state_dict.items():
            state_dict_data[key] = [[v.data for v in row] for row in mat]
        
        save_obj = {
            'config': self.config,
            'state_dict': state_dict_data,
            'vocab': self.tokenizer.vocab,
            'trained': self.trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_obj, f)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model weights from disk"""
        if not os.path.exists(path):
            return False
        
        with open(path, 'rb') as f:
            save_obj = pickle.load(f)
        
        self.config = save_obj['config']
        self.tokenizer = CharTokenizer(save_obj['vocab'])
        self.trained = save_obj['trained']
        
        # Convert raw data back to Value objects
        self.state_dict = {}
        for key, mat_data in save_obj['state_dict'].items():
            self.state_dict[key] = [[Value(v) for v in row] for row in mat_data]
        
        self.params = [p for mat in self.state_dict.values() for row in mat for p in row]
        print(f"Model loaded from {path}")
        return True

# ============================================================================
# Singleton instance for μSage
# ============================================================================

_model = None
_model_path = os.path.expanduser("~/.musage/minigpt_model.pkl")

def get_model():
    """Get or create the MiniGPT model"""
    global _model
    if _model is None:
        _model = MiniGPT()
        if os.path.exists(_model_path):
            _model.load(_model_path)
    return _model

def is_available():
    """Check if trained model is available"""
    return os.path.exists(_model_path)

def generate_answer(query, max_length=80):
    """Generate answer for a query"""
    model = get_model()
    if not model.trained:
        return None
    return model.generate(query, max_length=max_length, temperature=0.4)
